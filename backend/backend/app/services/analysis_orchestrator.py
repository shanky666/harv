"""
Analysis Orchestrator
Coordinates the new pipeline: Segmentation -> Background Removal -> Grading -> Reporting.
Replaces the old basket_analysis_service.
"""
import os
import sys
import uuid
import json
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.models import AnalysisSession, BasketFruit
from app.services.segmentation_service import segmentation_service
from app.services.background_removal import background_removal_service
from app.services.grading_service import grading_service
from app.services.vegetable_grading_service import vegetable_grading_service
from app.services.shelf_service import shelf_service
from app.services.market_service import market_service

BASE_PRICES = {
    "mango": 52.50,
    "grapes": 45.00,
    "pineapple": 85.00,
    "pomegranate": 40.00,
    "potato": 30.00,
    "carrot": 25.00,
    "tomato": 30.00,
    "onion": 20.00,
    "cucumber": 25.00,
    "capsicum": 40.00,
    "watermelon": 40.00,
    "banana": 35.00,
    "cocoa": 50.00,
    "coffee": 60.00,
    "strawberry": 25.00,
    "plum": 30.00,
    "peach": 35.00,
    "pear": 30.00,
}

QUALITY_MULTIPLIERS = {
    "Good": 1.0,
    "Better": 0.75,
    "Reject": 0.40,
}


def parse_price(price_range_str: str) -> float:
    import re
    nums = [int(s) for s in re.findall(r'\d+', price_range_str)]
    if not nums:
        return 50.0
    return sum(nums) / len(nums)


class AnalysisOrchestrator:
    """
    New analysis pipeline:

        User selects fruit type (dropdown)
        -> Upload image
        -> Segment individual fruits (watershed/contour)
        -> Remove background per fruit (mask-based)
        -> Grade each fruit (per-fruit MobileNetV2)
        -> Predict shelf life + market
        -> Persist to DB
        -> Generate report
    """

    def __init__(self):
        self._job_status: Dict[str, Dict[str, Any]] = {}

    def get_job_status(self, session_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        return self._job_status.get(str(session_id))

    async def run_analysis_job(
        self,
        image_path: str,
        session_id: uuid.UUID,
        fruit_type: str,
        is_single: bool = False,
        user_id: uuid.UUID = None,
    ) -> None:
        """Background task entry point."""
        from app.database.connection import AsyncSessionLocal

        self._job_status[str(session_id)] = {"status": "processing", "error": None}
        try:
            async with AsyncSessionLocal() as db:
                await self.analyze(
                    db, image_path, fruit_type,
                    is_single=is_single,
                    user_id=user_id,
                    session_id=session_id,
                )
            self._job_status[str(session_id)] = {"status": "complete", "error": None}
        except Exception as e:
            logger.exception(f"Analysis job failed for session {session_id}:")
            self._job_status[str(session_id)] = {"status": "failed", "error": str(e)}

    async def analyze(
        self,
        db: AsyncSession,
        image_path: str,
        fruit_type: str,
        is_single: bool = False,
        user_id: uuid.UUID = None,
        session_id: Optional[uuid.UUID] = None,
    ) -> Dict[str, Any]:
        """
        Runs the complete analysis pipeline.
        """
        import time

        start_all = time.time()
        session_id = session_id or uuid.uuid4()
        fruit_key = fruit_type.lower().strip()

        logger.info(
            f"Starting analysis session {session_id}: "
            f"fruit={fruit_type}, single={is_single}"
        )

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        h_img, w_img = image.shape[:2]

        seg_start = time.time()
        regions = segmentation_service.segment_fruits(
            image, fruit_type=fruit_key, is_single=is_single
        )
        seg_time = time.time() - seg_start
        logger.info(f"Segmentation completed: {len(regions)} fruit(s) in {seg_time:.2f}s")

        crop_start = time.time()
        crops_data = []
        for region in regions:
            crop_bgr = background_removal_service.extract_fruit_crop(
                image,
                region["mask"],
                region["bbox"],
                padding_ratio=0.05,
                target_size=(224, 224),
                refine_edges=False,
            )
            if crop_bgr is None:
                logger.warning(
                    f"Failed to extract crop for {region['fruit_id']}, skipping"
                )
                continue

            crop_path = os.path.join(
                "storage", "crops", str(session_id), f"{region['fruit_id']}.jpg"
            )
            os.makedirs(os.path.dirname(crop_path), exist_ok=True)
            cv2.imwrite(crop_path, crop_bgr)

            crops_data.append({
                "fruit_id": region["fruit_id"],
                "crop_bgr": crop_bgr,
                "crop_path": crop_path,
                "bbox": region["bbox"],
                "mask": region["mask"],
                "contour": region["contour"],
                "area": region["area"],
            })
        crop_time = time.time() - crop_start
        logger.info(f"Crop extraction: {len(crops_data)} crops in {crop_time:.2f}s")

        grade_start = time.time()
        grade_items = [
            {"fruit_id": cd["fruit_id"], "crop_bgr": cd["crop_bgr"]}
            for cd in crops_data
        ]
        _VEG = {"carrot", "tomato", "onion", "cucumber", "capsicum", "potato"}
        _is_veg = fruit_type.lower().strip() in _VEG
        _grading_svc = vegetable_grading_service if _is_veg else grading_service
        grade_results = _grading_svc.grade_batch(grade_items, fruit_type)
        grade_time = time.time() - grade_start
        logger.info(f"Grading completed in {grade_time:.2f}s")

        fruits_data = []
        summary = {"good": 0, "better": 0, "reject": 0}

        MIN_CONFIDENCE = 0.40

        for cd in crops_data:
            fid = cd["fruit_id"]
            grade_res = grade_results.get(fid)
            if grade_res is None:
                continue

            if grade_res["confidence"] < MIN_CONFIDENCE:
                logger.warning(
                    f"Dropping {fid}: confidence {grade_res['confidence']:.2%} "
                    f"< {MIN_CONFIDENCE:.0%} threshold"
                )
                continue

            grade = grade_res["grade"]
            grade_lower = grade.lower()
            defect_score = grade_res.get("defect_score", 0.0)

            if grade == "Mismatch":
                summary["reject"] = summary.get("reject", 0) + 1
                grade_color = "#E53935"
            elif grade_lower in summary:
                summary[grade_lower] += 1
                grade_color = "#66BB6A" if grade == "Good" else (
                    "#FFB74D" if grade == "Better" else "#E57373"
                )
            else:
                grade_color = "#E57373"

            rel_crop_path = f"storage/crops/{session_id}/{fid}.jpg"

            fruit_entry = {
                "fruit_id": fid,
                "fruit_type": fruit_type,
                "grade": grade,
                "confidence": grade_res["confidence"],
                "quality_confidence": grade_res["confidence"],
                "bbox": cd["bbox"],
                "crop_path": rel_crop_path,
                "local_crop_path": cd["crop_path"],
                "shelf_life_days": shelf_info["shelf_life_days"],
                "market_recommendation": market_info["recommended_market"],
                "grade_color": grade_color,
                "price": fruit_price,
                "defect_score": defect_score,
                "defects": grade_res.get("defects", []),
                "fruit_name": fruit_type.capitalize(),
                "quality": grade,
                "shelf_life": f"{shelf_info['shelf_life_days']} days",
                "market_value": fruit_price,
            }
            disease = grade_res.get("disease")
            if disease:
                fruit_entry["disease"] = disease
            fruits_data.append(fruit_entry)

        if not fruits_data:
            db_session = AnalysisSession(
                session_id=session_id,
                user_id=user_id,
                image_path=image_path,
                fruit_type=fruit_type,
                is_single=is_single,
                total_fruits=0,
                summary_json=json.dumps({"good": 0, "better": 0, "reject": 0}),
            )
            db.add(db_session)
            try:
                await db.commit()
                logger.info(f"Session {session_id} saved to database (no gradeable fruits)")
            except Exception as e:
                logger.exception("Failed to commit empty analysis to DB:")
                await db.rollback()

            total_time = time.time() - start_all
            logger.info(f"Analysis completed in {total_time:.2f}s (no gradeable fruits)")

            return self._build_response(
                session_id, fruits_data, 0,
                0, 0, 0,
                0.0, 0.0, 0.0,
                0, "Reject", 0.0,
                0.0, 0.0, 0.0,
                0.0, [],
                image_path, w_img, h_img,
            )

        total_count = len(fruits_data)
        good_count = sum(1 for f in fruits_data if f["grade"] == "Good")
        better_count = sum(1 for f in fruits_data if f["grade"] == "Better")
        reject_count = sum(1 for f in fruits_data if f["grade"] == "Reject")

        for fd in fruits_data:
            fd["count"] = total_count

        score_map = {"Good": 100, "Better": 75, "Reject": 30}
        score = round(
            sum(score_map.get(f["grade"], 75) for f in fruits_data) / total_count
        )
        if score >= 90:
            overall_grade = "Premium"
        elif score >= 70:
            overall_grade = "Good"
        else:
            overall_grade = "Reject"

        avg_shelf_life = round(
            sum(f["shelf_life_days"] for f in fruits_data) / total_count, 1
        )

        total_price = round(sum(f["price"] for f in fruits_data), 2)
        estimated_cost = total_price
        if overall_grade in ["Premium", "Good"]:
            estimated_selling_price = round(total_price * 1.30, 2)
            profit_estimation = round(estimated_selling_price - estimated_cost, 2)
        elif overall_grade in ["Better"]:
            estimated_selling_price = round(total_price * 0.90, 2)
            profit_estimation = 0.0
        else:
            estimated_selling_price = round(total_price * 0.20, 2)
            profit_estimation = 0.0

        good_pct = round((good_count / total_count) * 100, 1) if total_count > 0 else 0.0
        better_pct = round((better_count / total_count) * 100, 1) if total_count > 0 else 0.0
        reject_pct = round((reject_count / total_count) * 100, 1) if total_count > 0 else 0.0

        ai_recommendations = []
        if reject_count > 0:
            ai_recommendations.append(
                f"Discard rejected fruit(s) immediately to prevent cross-contamination."
            )
        if better_count > 0:
            ai_recommendations.append(
                f"Sell or consume {fruit_type.capitalize()} within 2 days or keep refrigerated."
            )
        if good_count > 0:
            ai_recommendations.append(
                f"Premium quality {fruit_type.capitalize()} detected. Suitable for export or high-end retail."
            )

        try:
            annotated_img = segmentation_service.draw_segmentation(
                image, regions, fruits_data
            )
            annotated_dir = os.path.join("storage", "uploads", "processed")
            os.makedirs(annotated_dir, exist_ok=True)
            annotated_path = os.path.join(annotated_dir, f"{session_id}_annotated.jpg")
            cv2.imwrite(annotated_path, annotated_img)
        except Exception as e:
            logger.warning(f"Failed to save annotated image: {e}")

        db_session = AnalysisSession(
            session_id=session_id,
            user_id=user_id,
            image_path=image_path,
            fruit_type=fruit_type,
            is_single=is_single,
            total_fruits=total_count,
            summary_json=json.dumps(summary),
        )
        db.add(db_session)

        for fd in fruits_data:
            bbox = fd.get("bbox", [0, 0, 0, 0])
            defects_list = fd.get("defects", [])
            defects_json = json.dumps(defects_list) if defects_list else None
            db_fruit = BasketFruit(
                id=uuid.uuid4(),
                session_id=session_id,
                fruit_id=fd["fruit_id"],
                fruit_type=fd["fruit_type"],
                grade=fd["grade"],
                grade_confidence=fd["confidence"],
                defect_score=fd.get("defect_score", 0.0),
                defects=defects_json,
                bbox_x1=float(bbox[0]) if bbox else None,
                bbox_y1=float(bbox[1]) if bbox else None,
                bbox_x2=float(bbox[2]) if bbox else None,
                bbox_y2=float(bbox[3]) if bbox else None,
                crop_path=fd["crop_path"],
                shelf_life=fd.get("shelf_life"),
                market_recommendation=fd.get("market_recommendation"),
            )
            db.add(db_fruit)

        try:
            await db.commit()
            logger.info(f"Session {session_id} saved to database")
        except Exception as e:
            logger.exception("Failed to commit analysis to DB:")
            await db.rollback()
            raise e

        total_time = time.time() - start_all
        logger.info(f"Analysis completed in {total_time:.2f}s")

        return self._build_response(
            session_id, fruits_data, total_count,
            good_count, better_count, reject_count,
            good_pct, better_pct, reject_pct,
            score, overall_grade, avg_shelf_life,
            total_price, estimated_cost, estimated_selling_price,
            profit_estimation, ai_recommendations,
            image_path, w_img, h_img,
        )

    def _build_response(self, session_id, fruits_data, total_count,
                        good_count, better_count, reject_count,
                        good_pct, better_pct, reject_pct,
                        score, overall_grade, avg_shelf_life,
                        total_price, estimated_cost, estimated_selling_price,
                        profit_estimation, ai_recommendations,
                        image_path, w_img, h_img):
        """Builds the API response payload."""
        fruit_type = fruits_data[0]["fruit_type"] if fruits_data else "unknown"

        return {
            "session_id": str(session_id),
            "total_fruits": total_count,
            "fruit_type": fruit_type,
            "fruits": [
                {
                    "fruit_id": f["fruit_id"],
                    "fruit_type": f["fruit_type"],
                    "grade": f["grade"],
                    "confidence": float(f["confidence"]),
                    "quality_confidence": f.get("quality_confidence", f["confidence"]),
                    "bbox": f["bbox"],
                    "crop_path": f["crop_path"],
                    "shelf_life_days": int(f["shelf_life_days"]),
                    "market_recommendation": f["market_recommendation"],
                    "grade_color": f.get("grade_color"),
                    "price": float(f.get("price", 0)),
                    "fruit_name": f.get("fruit_name"),
                    "count": f.get("count"),
                    "quality": f.get("quality"),
                    "shelf_life": f.get("shelf_life"),
                    "market_value": float(f.get("market_value", 0)),
                    "defect_score": float(f.get("defect_score", 0)),
                    "defects": f.get("defects", []),
                }
                for f in fruits_data
            ],
            "summary": {
                "good": good_count,
                "better": better_count,
                "reject": reject_count,
            },
            "demo_mode": False,
            "original_image_path": f"storage/uploads/original/{os.path.basename(image_path)}",
            "image_width": w_img,
            "image_height": h_img,
            "overall_grade": overall_grade,
            "score": score,
            "total_price": total_price,
            "estimated_selling_price": estimated_selling_price,
            "estimated_cost": estimated_cost,
            "profit_estimation": profit_estimation,
            "average_shelf_life": avg_shelf_life,
            "recommended_market": (
                "Export" if overall_grade == "Premium" else
                ("Supermarket" if overall_grade == "Good" else "Processing Industry")
            ),
            "ai_recommendations": ai_recommendations,
            "good_percentage": good_pct,
            "better_percentage": better_pct,
            "reject_percentage": reject_pct,
            "wholesale_value": f"Rs. {total_price:.2f}",
            "retail_value": f"Rs. {estimated_selling_price:.2f}",
            "estimated_profit": f"Rs. {profit_estimation:.2f}",
            "total_basket_value": f"Rs. {estimated_selling_price:.2f}",
        }

    async def get_analysis_history(
        self, db: AsyncSession, session_id: uuid.UUID
    ) -> Optional[Dict[str, Any]]:
        """Retrieves complete analysis history from DB."""
        query = select(AnalysisSession).where(
            AnalysisSession.session_id == session_id
        )
        result = await db.execute(query)
        db_session = result.scalars().first()
        if not db_session:
            return None

        query_fruits = select(BasketFruit).where(
            BasketFruit.session_id == session_id
        )
        result_fruits = await db.execute(query_fruits)
        db_fruits = result_fruits.scalars().all()

        try:
            summary = json.loads(db_session.summary_json)
        except Exception:
            summary = {"good": 0, "better": 0, "reject": 0}

        img = cv2.imread(db_session.image_path)
        h, w = img.shape[:2] if img is not None else (600, 800)

        fruits_payload = []
        for f in db_fruits:
            g_lower = f.grade.lower() if f.grade else "good"
            if g_lower in ["good"]:
                g_col = "#66BB6A"
            elif g_lower in ["better"]:
                g_col = "#FFB74D"
            else:
                g_col = "#E57373"

            fruit_lower = f.fruit_type.lower() if f.fruit_type else "mango"
            base_price = BASE_PRICES.get(fruit_lower, 50.0)
            multiplier = QUALITY_MULTIPLIERS.get(f.grade, 0.75)
            fruit_price = round(base_price * multiplier, 2)

            fruits_payload.append({
                "fruit_id": f.fruit_id,
                "fruit_type": f.fruit_type,
                "grade": f.grade,
                "grade_confidence": f.grade_confidence,
                "defect_score": f.defect_score,
                "defects": json.loads(f.defects) if f.defects else [],
                "bbox": [f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2] if f.bbox_x1 is not None else None,
                "crop_path": f.crop_path,
                "shelf_life": f.shelf_life or f"{shelf_service.predict_shelf_life(f.fruit_type, f.grade)['shelf_life_days']} days",
                "market_recommendation": f.market_recommendation or market_service.recommend_market(f.fruit_type, f.grade)["recommended_market"],
            })

        good_count = sum(1 for f in fruits_payload if f["grade"] == "Good")
        better_count = sum(1 for f in fruits_payload if f["grade"] == "Better")
        reject_count = sum(1 for f in fruits_payload if f["grade"] == "Reject")
        total_count = len(fruits_payload)

        score_map = {"Good": 100, "Better": 75, "Reject": 30}
        if total_count > 0:
            score = round(
                sum(score_map.get(f["grade"], 75) for f in fruits_payload) / total_count
            )
            overall_grade = "Premium" if score >= 90 else ("Good" if score >= 70 else "Reject")
            avg_shelf_life = round(
                sum(f.get("shelf_life_days", 3) for f in fruits_payload) / total_count, 1
            )
        else:
            score = 0
            overall_grade = "Reject"
            avg_shelf_life = 0.0

        total_price = 0
        for f in fruits_payload:
            fruit_lower = f["fruit_type"].lower() if f.get("fruit_type") else "mango"
            base_price = BASE_PRICES.get(fruit_lower, 50.0)
            multiplier = QUALITY_MULTIPLIERS.get(f["grade"], 0.75)
            total_price += round(base_price * multiplier, 2)
        total_price = round(total_price, 2)
        estimated_selling_price = round(total_price * 1.30, 2) if overall_grade in ["Premium", "Good"] else round(total_price * 0.20, 2)
        profit_estimation = round(estimated_selling_price - total_price, 2) if overall_grade in ["Premium", "Good"] else 0.0

        fruit_type = db_session.fruit_type or (fruits_payload[0]["fruit_type"] if fruits_payload else "unknown")

        ai_recommendations = []
        if reject_count > 0:
            ai_recommendations.append(
                f"Discard rejected fruit(s) immediately."
            )
        if better_count > 0:
            ai_recommendations.append(
                f"Consume {fruit_type.capitalize()} within 2 days."
            )
        if good_count > 0:
            ai_recommendations.append(
                f"Premium quality detected. Suitable for export."
            )

        return {
            "session_id": str(db_session.session_id),
            "total_fruits": total_count,
            "fruit_type": fruit_type,
            "is_single": db_session.is_single,
            "fruits": fruits_payload,
            "summary": summary,
            "original_image_path": f"storage/uploads/original/{os.path.basename(db_session.image_path)}",
            "image_width": w,
            "image_height": h,
            "overall_grade": overall_grade,
            "score": score,
            "total_price": total_price,
            "estimated_selling_price": estimated_selling_price,
            "average_shelf_life": avg_shelf_life,
            "recommended_market": (
                "Export" if overall_grade == "Premium" else
                ("Supermarket" if overall_grade == "Good" else "Processing Industry")
            ),
            "ai_recommendations": ai_recommendations,
        }


analysis_orchestrator = AnalysisOrchestrator()

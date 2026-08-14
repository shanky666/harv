import os
import uuid
import json
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete, select
from loguru import logger

from app.database.connection import get_db
from app.database.crud import (
    get_scan,
    bulk_create_fruits,
    update_scan_total,
    update_fruit_grade,
    create_passport,
    create_report,
    get_fruit_type_by_name,
    get_fruits_by_scan,
    get_report_by_scan,
    get_passport_by_fruit
)
from app.database.models import DetectedFruit, FruitPassport, Report, User
from app.database.schemas import ProcessResponse
from app.middleware.authentication import get_current_user

from app.services.yolo_service import yolo_service
from app.services.crop_service import crop_service
from app.services.grading_service import grading_service
from app.services.shelf_service import shelf_service
from app.services.market_service import market_service
from app.services.passport_service import passport_service
from app.services.report_service import report_service

router = APIRouter(prefix="/scan", tags=["Process"])

@router.post("/process/{scan_id}", response_model=ProcessResponse,
             summary="Run complete process pipeline (YOLO -> crop -> grade -> shelf-life -> market -> passport -> report)")
async def process_scan(
    scan_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    logger.info(f"🚀 Starting full pipeline processing for Scan: {scan_id}")
    
    # 1. Load Scan metadata
    scan = await get_scan(db, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
        
    # Clear any existing fruits, passports and reports for a clean rerun using CRUD helpers
    try:
        existing_report = await get_report_by_scan(db, scan_id)
        if existing_report:
            await db.delete(existing_report)
            
        existing_fruits = await get_fruits_by_scan(db, scan_id)
        for f in existing_fruits:
            passport = await get_passport_by_fruit(db, f.fruit_id)
            if passport:
                await db.delete(passport)
            await db.delete(f)
            
        await db.commit()
    except Exception as e:
        logger.error(f"Error clearing previous data: {e}")
        await db.rollback()


    # 2. Run YOLOv8 detection
    logger.info("Step 2: Running YOLOv8 detection...")
    detections = yolo_service.detect_fruits(scan.image_path, str(scan_id))
    
    # 3. Crop detected fruits
    logger.info("Step 3: Extracting ROI crops...")
    detections = crop_service.crop_fruits(scan.image_path, detections, str(scan_id))
    
    # 4. Save metadata for detected fruits in DB
    logger.info("Step 4: Persisting detected fruits...")
    db_fruits = []
    for det in detections:
        bb = det["bbox"]
        db_fruit_id = uuid.uuid4()
        
        db_fruit_type = await get_fruit_type_by_name(db, det["fruit_type"])
        fruit_type_id = db_fruit_type.id if db_fruit_type else None
        
        fruit = DetectedFruit(
            fruit_id=db_fruit_id,
            scan_id=scan_id,
            fruit_type_id=fruit_type_id,
            fruit_type=det["fruit_type"],
            bbox_x1=float(bb[0]), 
            bbox_y1=float(bb[1]),
            bbox_x2=float(bb[2]), 
            bbox_y2=float(bb[3]),
            confidence=det["confidence"],
            crop_path=det.get("crop_path")
        )
        db_fruits.append(fruit)

    await bulk_create_fruits(db, db_fruits)
    await update_scan_total(db, scan_id, len(db_fruits))
    
    # 5. Run grading, shelf life, and market recommendation for each fruit
    logger.info("Step 5: Grading and predicting post-harvest metrics...")
    fruit_dicts = []
    for f in db_fruits:
        grade_info = grading_service.grade_fruit(f.crop_path or "", f.fruit_type)
        grade = grade_info["grade"]
        confidence = grade_info["confidence"]
        defect_score = grade_info["defect_score"]
        defects = grade_info["defects"]
        predicted_at_str = grade_info["predicted_at"]
        predicted_at_dt = datetime.fromisoformat(predicted_at_str.replace("Z", "+00:00"))
        
        # Predict shelf life
        shelf_life_res = shelf_service.predict_shelf_life(f.fruit_type, grade, defect_score)
        shelf_life_days = shelf_life_res["shelf_life_days"]
        expiry_date = shelf_life_res["expiry_date"]
        shelf_life_str = f"{shelf_life_days} days"
        
        # Recommend market
        market_rec = market_service.recommend_market(f.fruit_type, grade, defect_score)
        recommended_market = market_rec["recommended_market"]
        estimated_price = market_rec["estimated_price"]
        
        # Save updates to DB
        await update_fruit_grade(
            db=db,
            fruit_id=f.fruit_id,
            grade=grade,
            grade_confidence=confidence,
            defect_score=defect_score,
            shelf_life=shelf_life_str,
            market=recommended_market,
            predicted_at=predicted_at_dt
        )
        
        # Resolve sequential fruit ID (e.g. FRUIT_0001) from path
        sequential_id = "FRUIT_UNKNOWN"
        if f.crop_path:
            sequential_id = os.path.splitext(os.path.basename(f.crop_path))[0]
            
        fruit_dicts.append({
            "fruit_id": sequential_id,
            "db_fruit_id": f.fruit_id,
            "fruit_type": f.fruit_type,
            "grade": grade,
            "confidence": confidence,
            "defect_score": defect_score,
            "defects": defects,
            "shelf_life_days": shelf_life_days,
            "expiry_date": expiry_date,
            "recommended_market": recommended_market,
            "estimated_price": estimated_price,
            "crop_path": f.crop_path or ""
        })

    # 6. Generate Fruit Passports
    logger.info("Step 6: Generating Fruit Passports...")
    for fd in fruit_dicts:
        defects_str = ", ".join(fd["defects"]) if fd["defects"] else "None"
        await create_passport(
            db=db,
            fruit_id=fd["db_fruit_id"],
            grade=fd["grade"],
            defects=defects_str,
            shelf_life=f"{fd['shelf_life_days']} days",
            market=fd["recommended_market"]
        )

    # 7. Generate Final Scan Report
    logger.info("Step 7: Generating final report...")
    report_data = report_service.generate_final_report(str(scan_id), fruit_dicts)
    
    # Save PDF and JSON summary in DB Report table
    grade_counts = {
        "good": report_data["good"],
        "better": report_data["better"],
        "medium": report_data["medium"],
        "reject": report_data["reject"]
    }
    
    agg_market = market_service.aggregate_recommendation(fruit_dicts)
    
    pdf_path = report_service.generate_pdf(
        str(scan_id),
        fruit_dicts,
        grade_counts,
        report_data["average_shelf_life"],
        agg_market
    )
    
    summary_json = json.dumps(report_data)
    await create_report(db, scan_id, pdf_path, summary_json)
    
    logger.info(f"✅ Full pipeline processing completed for Scan: {scan_id}")
    return report_data

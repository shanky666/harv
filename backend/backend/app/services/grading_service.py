"""
Fruit Quality Grading Service (Rewritten)
One shared MobileNetV2 architecture.
Four independent weight files.
No fruit type classification — the fruit is known from the dropdown.
"""
import os
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

from app.models.shared_mobilenet import (
    build_grading_model,
    preprocess_image,
    decode_prediction,
    FRUIT_CLASSES,
    FRUIT_NUM_CLASSES,
    VEGETABLE_BACKBONES,
)
from app.models.model_loader import load_grading_model, get_weight_path

DEFECT_OVERRIDE_DISABLED = {"orange", "guava", "kiwi", "banana", "potato", "cocoa", "coffee", "strawberry", "plum", "peach", "pear"}

FRUIT_V_THRESHOLDS = {
    "mango": 40,
    "pomegranate": 50,
    "pineapple": 60,
    "grapes": 45,
    "potato": 55,
    "banana": 45,
}


class GradingService:
    """
    Grades individual fruit crops using per-fruit MobileNetV2 models.

    The fruit type is known from the user dropdown — no classification step.
    Each fruit has its own weight file loaded at startup.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(GradingService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._models_cache: Dict[str, object] = {}
        self._initialized = True

    def load_models(self):
        """Pre-loads all available per-fruit grading models."""
        logger.info("Loading per-fruit grading models...")
        for fruit in ["mango", "pineapple", "grapes", "pomegranate", "orange", "guava", "kiwi", "watermelon", "banana", "cocoa", "coffee", "strawberry", "plum", "peach", "pear"]:
            model = load_grading_model(fruit)
            if model is not None:
                self._models_cache[fruit] = model
                path = get_weight_path(fruit)
                logger.info(f"  [{fruit.upper()}] loaded from {path}")
            else:
                logger.warning(f"  [{fruit.upper()}] no weights found — model unavailable")

        logger.info(
            f"Grading models loaded: {list(self._models_cache.keys())}"
        )

    def get_model(self, fruit_type: str) -> Optional[object]:
        """Returns the grading model for a specific fruit type."""
        return self._models_cache.get(fruit_type.lower().strip())

    def preprocess(self, crop_bgr: np.ndarray, backbone: str = "mobilenet") -> np.ndarray:
        return preprocess_image(crop_bgr, backbone=backbone)

    def grade_fruit(
        self,
        crop_bgr: np.ndarray,
        fruit_type: str,
        fruit_id: str = "",
    ) -> Dict[str, Any]:
        """
        Grades a single fruit crop.

        Args:
            crop_bgr: BGR crop image (cleaned, background-removed)
            fruit_type: Known fruit type from dropdown
            fruit_id: Optional fruit identifier

        Returns:
            {
                "grade": "Better" | "Good" | "Reject",
                "confidence": float,
                "defect_score": float,
                "defects": List[str],
                "predicted_at": str
            }
        """
        fruit_key = fruit_type.lower().strip()
        predicted_at = datetime.utcnow().isoformat() + "Z"

        features = self._extract_features(crop_bgr, fruit_key)
        defect_score = features["defect_score"]
        defects = features["visible_defects"]

        model = self.get_model(fruit_key)
        if model is None:
            logger.error(f"No model available for {fruit_key}. Cannot grade.")
            return {
                "grade": "Unavailable",
                "confidence": 0.0,
                "defect_score": defect_score,
                "defects": defects,
                "predicted_at": predicted_at,
                "error": f"No trained model found for '{fruit_key}'. Please train or upload a model.",
            }

        try:
            backbone = VEGETABLE_BACKBONES.get(fruit_key, "mobilenet")
            preprocessed = self.preprocess(crop_bgr, backbone=backbone)
            batch = np.expand_dims(preprocessed, axis=0)
            preds = model.predict(batch, verbose=0)[0]
            grade, confidence, disease_name = decode_prediction(preds, fruit_key)

            nc = FRUIT_NUM_CLASSES.get(fruit_key, 3)
            if nc == 2:
                log_str = f"Good={preds[0]:.4f} Reject={preds[1]:.4f}"
            else:
                top_idx = int(np.argmax(preds))
                top_name = disease_name if disease_name else FRUIT_CLASSES[top_idx] if top_idx < 3 else f"class_{top_idx}"
                top_conf = float(preds[top_idx])
                log_str = f"top={top_name}({top_conf:.4f})"

            logger.info(
                f"CNN preds [{fruit_id}]: {log_str} "
                f"| defect={defect_score:.4f}"
            )

            if nc <= 3 and fruit_key not in DEFECT_OVERRIDE_DISABLED:
                if nc == 2:
                    reject_prob = float(preds[1])
                    best_non_reject = float(preds[0])
                else:
                    reject_prob = float(preds[2])
                    best_non_reject = float(np.max(preds[:2]))
                margin = best_non_reject - reject_prob
                if defect_score > 0.7:
                    grade = "Reject"
                    confidence = round(min(reject_prob + 0.1, 0.99), 4)
                    logger.info(f"Reject by defect override: score={defect_score:.4f}")
                elif margin < 0.2 and defect_score > 0.5:
                    grade = "Reject"
                    confidence = round(reject_prob, 4)
                    logger.info(f"Reject by uncertainty+defect: margin={margin:.4f}, defect={defect_score:.4f}")

            result = {
                "grade": grade,
                "confidence": round(confidence, 4),
                "defect_score": defect_score,
                "defects": defects,
                "predicted_at": predicted_at,
            }
            if disease_name:
                result["disease"] = disease_name
            return result

        except Exception as e:
            logger.error(f"CNN inference failed for {fruit_key}: {e}")
            return {
                "grade": "Unavailable",
                "confidence": 0.0,
                "defect_score": defect_score,
                "defects": defects,
                "predicted_at": predicted_at,
                "error": f"Model inference failed: {str(e)}",
            }

    def grade_batch(
        self,
        items: List[Dict[str, Any]],
        fruit_type: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Grades multiple crops of the SAME fruit type in one batched call.

        Args:
            items: [{"fruit_id": str, "crop_bgr": np.ndarray}, ...]
            fruit_type: The fruit type for all items

        Returns:
            {fruit_id: {grade, confidence, defect_score, defects, predicted_at}}
        """
        results: Dict[str, Dict[str, Any]] = {}
        predicted_at = datetime.utcnow().isoformat() + "Z"
        fruit_key = fruit_type.lower().strip()

        model = self.get_model(fruit_key)

        valid_items = []
        features_by_id = {}

        for item in items:
            crop_bgr = item.get("crop_bgr")
            fid = item.get("fruit_id", "")
            if crop_bgr is None or crop_bgr.size == 0:
                logger.warning(f"Skipping empty crop for {fid}")
                continue
            features_by_id[fid] = self._extract_features(crop_bgr, fruit_key)
            valid_items.append(item)

        if model is None:
            logger.error(f"No model available for {fruit_key}. Cannot grade batch.")
            for it in valid_items:
                fid = it["fruit_id"]
                defect_score = features_by_id[fid]["defect_score"]
                defects = features_by_id[fid]["visible_defects"]
                results[fid] = {
                    "grade": "Unavailable",
                    "confidence": 0.0,
                    "defect_score": defect_score,
                    "defects": defects,
                    "predicted_at": predicted_at,
                    "error": f"No trained model found for '{fruit_key}'.",
                }
            return results

        try:
            backbone = VEGETABLE_BACKBONES.get(fruit_key, "mobilenet")
            batch = np.stack(
                [self.preprocess(it["crop_bgr"], backbone=backbone) for it in valid_items], axis=0
            )
            preds = model.predict(batch, verbose=0)

            for it, pred in zip(valid_items, preds):
                fid = it["fruit_id"]
                defect_score = features_by_id[fid]["defect_score"]
                defects = features_by_id[fid]["visible_defects"]

                grade, confidence, disease_name = decode_prediction(pred, fruit_key)

                nc = FRUIT_NUM_CLASSES.get(fruit_key, 3)
                if nc <= 3 and fruit_key not in DEFECT_OVERRIDE_DISABLED:
                    if nc == 2:
                        reject_prob = float(pred[1])
                        best_non_reject = float(pred[0])
                    else:
                        reject_prob = float(pred[2])
                        best_non_reject = float(np.max(pred[:2]))
                    margin = best_non_reject - reject_prob
                    if defect_score > 0.7:
                        grade = "Reject"
                        confidence = round(min(reject_prob + 0.1, 0.99), 4)
                    elif margin < 0.2 and defect_score > 0.5:
                        grade = "Reject"
                        confidence = round(reject_prob, 4)

                result = {
                    "grade": grade,
                    "confidence": round(confidence, 4),
                    "defect_score": defect_score,
                    "defects": defects,
                    "predicted_at": predicted_at,
                }
                if disease_name:
                    result["disease"] = disease_name
                results[fid] = result

            return results

        except Exception as e:
            logger.error(f"Batched inference failed for {fruit_key}: {e}")
            for it in valid_items:
                fid = it["fruit_id"]
                defect_score = features_by_id[fid]["defect_score"]
                defects = features_by_id[fid]["visible_defects"]
                results[fid] = {
                    "grade": "Unavailable",
                    "confidence": 0.0,
                    "defect_score": defect_score,
                    "defects": defects,
                    "predicted_at": predicted_at,
                    "error": f"Model inference failed: {str(e)}",
                }
            return results


    def _extract_features(
        self, crop_bgr: np.ndarray, fruit_type: str = "default"
    ) -> Dict[str, Any]:
        """
        Extracts color, texture, shape, and defect features using OpenCV.
        Used for defect scoring (passed through in results for diagnostics).
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return {
                "color_intensity": 0.5,
                "texture_roughness": 0.5,
                "aspect_ratio": 1.0,
                "defect_score": 0.0,
                "visible_defects": [],
            }

        h, w = crop_bgr.shape[:2]

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        mean_hsv = cv2.mean(hsv)
        sat = mean_hsv[1]

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        std_dev = float(np.std(gray))
        texture_roughness = min(1.0, std_dev / 128.0)

        aspect_ratio = float(w) / float(h) if h > 0 else 1.0

        v_channel = hsv[:, :, 2]
        dark_thresh = FRUIT_V_THRESHOLDS.get(fruit_type, 50)
        _, dark_spots = cv2.threshold(v_channel, dark_thresh, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_spots = cv2.morphologyEx(dark_spots, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(
            dark_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        defect_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > 200:
                defect_area += area

        total_area = float(h * w)
        defect_ratio = defect_area / total_area if total_area > 0 else 0.0
        spot_score = min(1.0, defect_ratio * 6.0)

        hue = hsv[:, :, 0].astype(float)
        hue_std = float(np.std(hue)) / 180.0
        sat_img = hsv[:, :, 1].astype(float)
        sat_std = float(np.std(sat_img)) / 255.0
        color_uniformity = min(1.0, (hue_std + sat_std) * 0.5)

        defect_score = min(1.0, spot_score * 0.9 + color_uniformity * 0.1)

        visible_defects = []
        if defect_score > 0.4:
            visible_defects.append("severe_rot_or_bruising")
        elif defect_score > 0.15:
            visible_defects.append("moderate_surface_blemishes")
        elif defect_score > 0.03:
            visible_defects.append("minor_spots")

        if aspect_ratio < 0.7 or aspect_ratio > 1.4:
            visible_defects.append("shape_deformation")
            defect_score = min(1.0, defect_score + 0.2)

        return {
            "color_intensity": round(float(sat / 255.0), 4),
            "texture_roughness": round(texture_roughness, 4),
            "aspect_ratio": round(aspect_ratio, 4),
            "defect_score": round(defect_score, 4),
            "visible_defects": visible_defects,
        }


grading_service = GradingService()

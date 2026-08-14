"""
Vegetable Quality Grading Service (2-class only)
Separate from fruit grading. Always Good/Reject.
EfficientNetB0 architecture (better accuracy than MobileNetV2 used for fruits).
"""
import os
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

from app.models.shared_mobilenet import (
    build_grading_model,
    build_vegetable_model,
    preprocess_image,
    decode_prediction,
    FRUIT_NUM_CLASSES,
)

VEGETABLE_V_THRESHOLDS = {
    "carrot": 55,
    "tomato": 60,
    "potato": 55,
    "onion": 60,
    "cucumber": 50,
    "capsicum": 45,
}

DEFECT_OVERRIDE_DISABLED = {"capsicum", "carrot", "potato"}


def get_vegetable_weight_path(veg: str) -> str | None:
    base = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.abspath(os.path.join(base, ".."))
    path = os.path.join(parent, "models", "weights", f"{veg}.keras")
    if os.path.exists(path):
        return path
    return None


def load_vegetable_model(veg: str):
    path = get_vegetable_weight_path(veg)
    if path is None:
        return None
    try:
        model = build_vegetable_model(num_classes=2)
        model.load_weights(path)
        logger.info(f"Loaded {veg} model from {path} (num_classes=2, backbone=efficientnet)")
        return model
    except Exception as e:
        logger.error(f"Failed to load {veg} model: {e}")
        return None


class VegetableGradingService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VegetableGradingService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._models_cache: Dict[str, object] = {}
        self._initialized = True

    def load_models(self, vegetables: List[str] = None):
        if vegetables is None:
            vegetables = []
        logger.info("Loading vegetable grading models...")
        for veg in vegetables:
            model = load_vegetable_model(veg)
            if model is not None:
                self._models_cache[veg] = model
                logger.info(f"  [{veg.upper()}] loaded")
            else:
                logger.warning(f"  [{veg.upper()}] no weights — heuristic fallback")
        logger.info(f"Vegetable models loaded: {list(self._models_cache.keys())}")

    def get_model(self, veg_type: str) -> Optional[object]:
        return self._models_cache.get(veg_type.lower().strip())

    def preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        return preprocess_image(crop_bgr, backbone="efficientnet")

    def grade_vegetable(
        self,
        crop_bgr: np.ndarray,
        veg_type: str,
        veg_id: str = "",
    ) -> Dict[str, Any]:
        veg_key = veg_type.lower().strip()
        predicted_at = datetime.utcnow().isoformat() + "Z"

        features = self._extract_features(crop_bgr, veg_key)
        defect_score = features["defect_score"]
        defects = features["visible_defects"]

        model = self.get_model(veg_key)
        if model is not None:
            try:
                preprocessed = self.preprocess(crop_bgr)
                batch = np.expand_dims(preprocessed, axis=0)
                preds = model.predict(batch, verbose=0)[0]
                grade, confidence, disease_name = decode_prediction(preds, veg_key)

                reject_prob = float(preds[1])
                best_non_reject = float(preds[0])
                margin = best_non_reject - reject_prob

                logger.info(
                    f"Veg CNN preds [{veg_id}]: Good={preds[0]:.4f} Reject={preds[1]:.4f} "
                    f"| margin={margin:.4f} defect={defect_score:.4f}"
                )

                if veg_key not in DEFECT_OVERRIDE_DISABLED:
                    if defect_score > 0.7:
                        grade = "Reject"
                        confidence = round(min(reject_prob + 0.1, 0.99), 4)
                        logger.info(
                            f"Veg Reject by defect override: score={defect_score:.4f} "
                            f"(CNN had predicted {grade})"
                        )
                    elif margin < 0.2 and defect_score > 0.5:
                        grade = "Reject"
                        confidence = round(reject_prob, 4)
                        logger.info(
                            f"Veg Reject by uncertainty+defect: margin={margin:.4f}, "
                            f"defect={defect_score:.4f}"
                        )

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
                logger.error(f"CNN inference failed for {veg_key}: {e}")

        return self._heuristic_grade(defect_score, defects, predicted_at)

    def grade_batch(
        self,
        items: List[Dict[str, Any]],
        veg_type: str,
    ) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        predicted_at = datetime.utcnow().isoformat() + "Z"
        veg_key = veg_type.lower().strip()

        model = self.get_model(veg_key)

        valid_items = []
        features_by_id = {}

        for item in items:
            crop_bgr = item.get("crop_bgr")
            fid = item.get("fruit_id", "")
            if crop_bgr is None or crop_bgr.size == 0:
                logger.warning(f"Skipping empty crop for {fid}")
                continue
            features_by_id[fid] = self._extract_features(crop_bgr, veg_key)
            valid_items.append(item)

        if model is not None and valid_items:
            try:
                batch = np.stack(
                    [self.preprocess(it["crop_bgr"]) for it in valid_items], axis=0
                )
                preds = model.predict(batch, verbose=0)

                for it, pred in zip(valid_items, preds):
                    fid = it["fruit_id"]
                    defect_score = features_by_id[fid]["defect_score"]
                    defects = features_by_id[fid]["visible_defects"]

                    grade, confidence, disease_name = decode_prediction(pred, veg_key)
                    reject_prob = float(pred[1])
                    best_non_reject = float(pred[0])
                    margin = best_non_reject - reject_prob

                    if veg_key not in DEFECT_OVERRIDE_DISABLED:
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
                logger.error(f"Batched veg inference failed for {veg_key}: {e}")

        for it in valid_items:
            fid = it["fruit_id"]
            defect_score = features_by_id[fid]["defect_score"]
            defects = features_by_id[fid]["visible_defects"]
            result = self._heuristic_grade(defect_score, defects, predicted_at)
            results[fid] = result

        return results

    def _heuristic_grade(
        self,
        defect_score: float,
        defects: List[str],
        predicted_at: str,
    ) -> Dict[str, Any]:
        if defect_score > 0.4:
            grade, confidence = "Reject", 0.70
        elif defect_score > 0.15:
            grade, confidence = "Good", 0.65
        else:
            grade, confidence = "Good", 0.75

        return {
            "grade": grade,
            "confidence": confidence,
            "defect_score": defect_score,
            "defects": defects,
            "predicted_at": predicted_at,
        }

    def _extract_features(
        self, crop_bgr: np.ndarray, veg_type: str = "default"
    ) -> Dict[str, Any]:
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
        mean_hsv = cv2.mean(hsv)
        sat = mean_hsv[1]

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        std_dev = float(np.std(gray))
        texture_roughness = min(1.0, std_dev / 128.0)

        aspect_ratio = float(w) / float(h) if h > 0 else 1.0

        v_channel = hsv[:, :, 2]
        dark_thresh = VEGETABLE_V_THRESHOLDS.get(veg_type, 50)
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


vegetable_grading_service = VegetableGradingService()

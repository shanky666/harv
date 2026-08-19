"""
Fruit Type Classification & Verification Service
Verifies that an uploaded crop matches the expected fruit selected in the dropdown.
Uses deep learning model predictions when available, backed by OpenCV visual feature verification.
"""
import os
import cv2
import json
import numpy as np
from typing import Dict, Any, Optional, Tuple
from loguru import logger

KNOWN_FRUITS = [
    "grapes", "mango", "pineapple", "pomegranate",
    "orange", "guava", "banana", "strawberry",
    "kiwi", "watermelon", "potato", "carrot", "tomato"
]

FRUIT_COLOR_PROFILES = {
    "grapes": {
        "aspect_ratio_range": (0.6, 1.5),
        "typical_hue_ranges": [(25, 85), (120, 165)],  # green & purple/violet
        "min_saturation": 30,
        "max_size_ratio": 0.45,
    },
    "pineapple": {
        "aspect_ratio_range": (0.4, 2.2),
        "typical_hue_ranges": [(10, 40), (40, 90)],   # golden brown/yellow & green crown
        "min_saturation": 40,
        "textured": True,
    },
    "banana": {
        "aspect_ratio_range": (1.6, 4.5),             # elongated
        "typical_hue_ranges": [(15, 60)],              # yellow/green
        "min_saturation": 40,
    },
    "orange": {
        "aspect_ratio_range": (0.75, 1.35),           # round
        "typical_hue_ranges": [(5, 28)],               # orange
        "min_saturation": 60,
    },
    "pomegranate": {
        "aspect_ratio_range": (0.75, 1.35),           # round
        "typical_hue_ranges": [(0, 18), (160, 180)],   # deep red / brownish
        "min_saturation": 40,
    },
    "mango": {
        "aspect_ratio_range": (0.7, 1.8),
        "typical_hue_ranges": [(10, 60)],              # yellow/red/green
        "min_saturation": 35,
    },
    "strawberry": {
        "aspect_ratio_range": (0.6, 1.6),
        "typical_hue_ranges": [(0, 15), (165, 180)],   # bright red
        "min_saturation": 50,
    },
    "guava": {
        "aspect_ratio_range": (0.7, 1.4),
        "typical_hue_ranges": [(30, 85)],              # green / yellow-green
        "min_saturation": 30,
    },
}


class ClassificationService:
    """
    Handles fruit type classification and verification to prevent fruit mismatches.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ClassificationService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self.model = None
        self.classes = KNOWN_FRUITS
        self._initialized = True
        self._load_model()

    def _load_model(self):
        """Attempts to load a trained fruit type classification model."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.abspath(os.path.join(base_dir, "..", ".."))

        candidates = [
            os.path.join(backend_dir, "models", "fruit_classifier.h5"),
            os.path.join(backend_dir, "app", "models", "weights", "fruit_classifier.keras"),
        ]

        for path in candidates:
            if os.path.exists(path):
                try:
                    import keras
                    self.model = keras.models.load_model(path, compile=False)
                    logger.info(f"ClassificationService loaded model from {path}")

                    # Load class indices mapping if exists
                    indices_path = os.path.join(os.path.dirname(path), "class_indices.json")
                    if os.path.exists(indices_path):
                        with open(indices_path, "r") as f:
                            mapping = json.load(f)
                            # Sort by index
                            self.classes = [k for k, v in sorted(mapping.items(), key=lambda item: item[1])]
                    return
                except Exception as e:
                    logger.warning(f"Failed to load classification model at {path}: {e}")

        logger.info("ClassificationService operating with OpenCV feature profile verification")

    def classify_fruit(self, image_or_crop: Any) -> Dict[str, Any]:
        """
        Classifies an image crop into one of the known fruit classes.
        """
        if isinstance(image_or_crop, str):
            img_bgr = cv2.imread(image_or_crop)
        else:
            img_bgr = image_or_crop

        if img_bgr is None or img_bgr.size == 0:
            return {"fruit_type": "unknown", "confidence": 0.0}

        # Use DL Model if available
        if self.model is not None:
            try:
                from app.models.shared_mobilenet import preprocess_image
                preprocessed = preprocess_image(img_bgr, backbone="mobilenet")
                batch = np.expand_dims(preprocessed, axis=0)
                preds = self.model.predict(batch, verbose=0)[0]
                idx = int(np.argmax(preds))
                predicted_class = self.classes[idx] if idx < len(self.classes) else "unknown"
                confidence = float(preds[idx])
                return {"fruit_type": predicted_class, "confidence": round(confidence, 4)}
            except Exception as e:
                logger.error(f"Classification model inference error: {e}")

        # Fallback to feature signature identification
        detected_fruit, confidence = self._feature_based_classify(img_bgr)
        return {"fruit_type": detected_fruit, "confidence": confidence}

    def verify_fruit(self, crop_bgr: Any, expected_fruit: str) -> Dict[str, Any]:
        """
        Verifies if the given crop matches the expected fruit selected by the user.

        Returns:
            {
                "is_match": bool,
                "predicted_fruit": str,
                "confidence": float,
                "warning": Optional[str]
            }
        """
        if isinstance(crop_bgr, str):
            crop_bgr = cv2.imread(crop_bgr)

        if crop_bgr is None or crop_bgr.size == 0:
            return {
                "is_match": True,
                "predicted_fruit": expected_fruit,
                "confidence": 0.5,
                "warning": None,
            }

        expected_key = expected_fruit.lower().strip()
        classification = self.classify_fruit(crop_bgr)
        predicted_fruit = classification["fruit_type"]
        confidence = classification["confidence"]

        # 1. Check DL classification match
        if predicted_fruit != "unknown" and confidence > 0.60:
            if predicted_fruit == expected_key:
                return {
                    "is_match": True,
                    "predicted_fruit": predicted_fruit,
                    "confidence": confidence,
                    "warning": None,
                }
            else:
                warning_msg = (
                    f"Image Mismatch: Uploaded image appears to be a '{predicted_fruit.capitalize()}', "
                    f"but '{expected_fruit.capitalize()}' was selected in the dropdown."
                )
                logger.warning(warning_msg)
                return {
                    "is_match": False,
                    "predicted_fruit": predicted_fruit,
                    "confidence": confidence,
                    "warning": warning_msg,
                }

        # 2. Check Feature Profile Compatibility
        is_profile_match, detected_profile, profile_warning = self._check_feature_profile(crop_bgr, expected_key)
        if not is_profile_match:
            return {
                "is_match": False,
                "predicted_fruit": detected_profile or "Different Fruit",
                "confidence": 0.85,
                "warning": profile_warning,
            }

        return {
            "is_match": True,
            "predicted_fruit": expected_key,
            "confidence": 0.90,
            "warning": None,
        }

    def _feature_based_classify(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Heuristic identification based on aspect ratio, color spectrum, and texture."""
        h, w = crop_bgr.shape[:2]
        if h == 0 or w == 0:
            return "unknown", 0.0

        aspect_ratio = float(w) / float(h)
        if aspect_ratio < 1.0:
            aspect_ratio = 1.0 / aspect_ratio  # Normalized ratio >= 1.0

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        mean_hue = float(np.mean(hsv[:, :, 0]))
        mean_sat = float(np.mean(hsv[:, :, 1]))
        mean_val = float(np.mean(hsv[:, :, 2]))

        # Banana check (very elongated)
        if aspect_ratio > 1.8 and 15 <= mean_hue <= 60:
            return "banana", 0.80

        # Pineapple check (golden/brownish hue or distinct crown/textured skin)
        if 10 <= mean_hue <= 40 and mean_val > 50 and mean_sat > 40:
            # Check for high texture/variance
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            if np.std(gray) > 35:
                return "pineapple", 0.82

        # Orange check (strong orange saturation)
        if 5 <= mean_hue <= 25 and mean_sat > 100:
            return "orange", 0.80

        # Strawberry / Pomegranate check (red)
        if (mean_hue < 12 or mean_hue > 165) and mean_sat > 70:
            return "strawberry" if aspect_ratio < 1.3 else "pomegranate", 0.75

        # Default fallback to expected
        return "unknown", 0.50

    def _check_feature_profile(self, crop_bgr: np.ndarray, expected_key: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Validates crop features against the expected fruit's visual profile."""
        if expected_key not in FRUIT_COLOR_PROFILES:
            return True, None, None

        profile = FRUIT_COLOR_PROFILES[expected_key]
        h, w = crop_bgr.shape[:2]
        if h == 0 or w == 0:
            return True, None, None

        aspect_ratio = float(w) / float(h)
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        mean_hue = float(np.mean(hsv[:, :, 0]))
        mean_sat = float(np.mean(hsv[:, :, 1]))

        # Check Grapes vs Pineapple/Banana mismatch
        if expected_key == "grapes":
            # Pineapple profile check: golden-brown hue (10-40) with high texture or aspect ratio mismatch
            if 12 <= mean_hue <= 40 and mean_sat > 50:
                gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
                if np.std(gray) > 35:
                    msg = "Image Mismatch: Uploaded image appears to be a Pineapple, but Grapes was selected in the dropdown."
                    return False, "pineapple", msg
            # Banana profile check
            if aspect_ratio > 1.95 or aspect_ratio < 0.5:
                msg = "Image Mismatch: Uploaded image shape does not match Grapes."
                return False, "banana", msg

        # Check Pineapple vs Grapes mismatch
        if expected_key == "pineapple":
            if (45 <= mean_hue <= 85 or mean_hue > 120) and aspect_ratio < 1.3:
                gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
                if np.std(gray) < 25:
                    msg = "Image Mismatch: Uploaded image appears to be Grapes or Guava, but Pineapple was selected."
                    return False, "grapes", msg

        return True, None, None


classification_service = ClassificationService()

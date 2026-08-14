"""
Model Loader
Dynamically loads per-fruit grading weight files at runtime.
"""
import os
from typing import Optional, Dict
from loguru import logger

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

_model_cache: Dict[str, object] = {}


def _resolve_weight_path(fruit: str) -> str:
    """
    Resolves the weight file path for a given fruit.

    Search order:
        1. models/weights/{fruit}.keras
        2. models/weights/{fruit}.h5
        3. models/{fruit}_quality_ft.h5 (legacy)
        4. models/{fruit}_model.h5 (legacy)
        5. app/ai/models/{fruit}_model.h5 (legacy)
    """
    fruit = fruit.lower().strip()
    service_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.abspath(os.path.join(service_dir, "..", ".."))

    candidates = [
        os.path.join(WEIGHTS_DIR, f"{fruit}.keras"),
        os.path.join(WEIGHTS_DIR, f"{fruit}.h5"),
        os.path.join(backend_dir, "models", f"{fruit}_quality_ft.h5"),
        os.path.join(backend_dir, "models", f"{fruit}_model.h5"),
        os.path.join(backend_dir, "app", "ai", "models", f"{fruit}_model.h5"),
        os.path.join(backend_dir, "..", "models", f"{fruit}_quality_ft.h5"),
    ]

    for path in candidates:
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            if size_kb < 50:
                logger.warning(f"Skipping placeholder weights for {fruit}: {path} ({size_kb:.0f} KB)")
                continue
            return path

    return ""


def load_grading_model(fruit: str) -> Optional[object]:
    """
    Loads the grading model for a specific fruit.

    Returns the Keras model or None if not available.
    Models are cached in memory after first load.
    """
    fruit = fruit.lower().strip()

    if fruit in _model_cache:
        return _model_cache[fruit]

    weight_path = _resolve_weight_path(fruit)
    if not weight_path:
        logger.warning(f"No grading weights found for {fruit}")
        return None

    try:
        import keras
        model = keras.models.load_model(weight_path, compile=False)
        logger.info(f"Loaded grading model for {fruit} from {weight_path}")
        _model_cache[fruit] = model
        return model
    except Exception as e:
        logger.error(f"Failed to load grading model for {fruit}: {e}")
        return None


def get_weight_path(fruit: str) -> Optional[str]:
    """Returns the resolved weight path without loading the model."""
    return _resolve_weight_path(fruit) or None


def clear_cache():
    """Clears the model cache (useful for hot-reloading)."""
    _model_cache.clear()

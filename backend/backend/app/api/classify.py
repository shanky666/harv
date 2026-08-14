"""
Fruit Classification Debug API
Accepts a single fruit image and returns the predicted fruit type with confidence.
Useful for testing the MobileNetV2 classifier independently of the full pipeline.
"""
import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from loguru import logger

from app.core.config import settings
from app.services.classification_service import classification_service
from app.middleware.authentication import get_current_user_demo
from app.database.models import User
from typing import Optional

router = APIRouter(prefix="/scan", tags=["Classification"])


@router.post(
    "/classify",
    summary="Classify a single fruit image using the MobileNetV2 fruit classifier"
)
async def classify_single_fruit(
    file: UploadFile = File(...),
    current_user: Optional[User] = Depends(get_current_user_demo)
):
    """
    Upload a single cropped or whole fruit image.
    Returns the predicted fruit type and confidence score from the MobileNetV2 classifier.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    # Save to a temp path in storage/crops/classify/
    safe_name = "".join(c for c in os.path.basename(file.filename or "upload.jpg") if c.isalnum() or c in "._-")
    tmp_filename = f"{uuid.uuid4()}_{safe_name}"
    tmp_dir = os.path.join(settings.STORAGE_BASE_PATH, "crops", "classify")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, tmp_filename)

    try:
        with open(tmp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
        logger.info(f"Saved classify upload to: {tmp_path}")
    except Exception as e:
        logger.error(f"Failed to save upload for classification: {e}")
        raise HTTPException(status_code=500, detail="Could not save uploaded image.")

    try:
        result = classification_service.classify_fruit(tmp_path)
        model_status = "loaded" if classification_service.model is not None else "heuristic"
        return {
            "filename": safe_name,
            "fruit_type": result.get("fruit_type", "unknown"),
            "confidence": result.get("confidence", 0.0),
            "classifier_status": model_status,
            "classifier_classes": classification_service.classes,
        }
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Classification error: {e}")
    finally:
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass

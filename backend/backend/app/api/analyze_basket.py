"""
Analysis API Router
New endpoint: user selects fruit type, uploads image, gets quality analysis.
"""
import os
import uuid
import shutil
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger

from app.core.config import settings
from app.database.connection import get_db
from app.database.schemas import BasketAnalysisResponse
from app.services.analysis_orchestrator import analysis_orchestrator
from app.services.vegetable_grading_service import vegetable_grading_service

router = APIRouter(tags=["Analysis"])

SUPPORTED_FRUITS = ["mango", "pineapple", "grapes", "pomegranate", "orange", "guava", "kiwi", "watermelon", "banana", "cocoa", "coffee", "strawberry", "plum", "peach", "pear"]
SUPPORTED_VEGETABLES = ["carrot", "tomato", "onion", "cucumber", "capsicum", "potato"]


@router.post(
    "/analyze",
    summary="Analyze fruit quality (user selects fruit type)",
)
async def analyze_fruit(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    fruit_type: str = Query(..., description="Fruit/veg type: mango, pineapple, grapes, pomegranate, banana, orange, guava, etc."),
    is_single: bool = Query(True, description="True for single fruit, False for basket"),
    db: AsyncSession = Depends(get_db),
):
    """
    New analysis endpoint.

    1. User selects fruit type from dropdown
    2. User selects mode (single fruit or basket)
    3. User uploads image
    4. Backend segments, removes background, grades each fruit
    5. Returns quality report

    No fruit type classification is performed.
    """
    fruit_key = fruit_type.lower().strip()
    all_supported = SUPPORTED_FRUITS + SUPPORTED_VEGETABLES
    if fruit_key not in all_supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type: {fruit_type}. "
                   f"Fruits: {', '.join(SUPPORTED_FRUITS)} | "
                   f"Vegetables: {', '.join(SUPPORTED_VEGETABLES)}"
        )

    logger.info(f"Analysis request: type={fruit_type}, single={is_single}")

    safe_filename = "".join(
        c for c in os.path.basename(file.filename) if c.isalnum() or c in "._- "
    )
    original_filename = f"{uuid.uuid4()}_{safe_filename}"
    dest_dir = os.path.join(settings.STORAGE_BASE_PATH, "uploads", "original")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, original_filename)

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved upload to: {dest_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded image: {e}")
        raise HTTPException(status_code=500, detail="Could not save uploaded image.")

    session_id = uuid.uuid4()

    background_tasks.add_task(
        analysis_orchestrator.run_analysis_job,
        dest_path,
        session_id,
        fruit_type,
        is_single,
    )

    return {"session_id": str(session_id), "status": "processing", "fruit_type": fruit_type}


@router.get(
    "/analysis/{session_id}/status",
    summary="Poll analysis job status",
)
async def get_analysis_status(session_id: uuid.UUID):
    """Lightweight polling endpoint for background job status."""
    job = analysis_orchestrator.get_job_status(session_id)
    if job is None:
        return {"status": "unknown"}
    return job


@router.get(
    "/analysis/{session_id}",
    response_model=BasketAnalysisResponse,
    summary="Retrieve full analysis results",
)
async def get_analysis(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Returns the complete analysis results for a session."""
    try:
        history = await analysis_orchestrator.get_analysis_history(db, session_id)
        if not history:
            raise HTTPException(status_code=404, detail="Analysis session not found.")
        return history
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to query analysis.")


@router.get("/fruits/supported", summary="List supported fruit and vegetable types")
async def list_supported_fruits():
    """Returns the list of supported types for the dropdown."""
    return {
        "fruits": [
            {"key": "mango", "name": "Mango", "scientific": "Mangifera indica"},
            {"key": "pineapple", "name": "Pineapple", "scientific": "Ananas comosus"},
            {"key": "grapes", "name": "Grapes", "scientific": "Vitis vinifera"},
            {"key": "pomegranate", "name": "Pomegranate", "scientific": "Punica granatum"},
            {"key": "orange", "name": "Orange", "scientific": "Citrus sinensis"},
            {"key": "guava", "name": "Guava", "scientific": "Psidium guajava"},
            {"key": "kiwi", "name": "Kiwi", "scientific": "Actinidia deliciosa"},
            {"key": "watermelon", "name": "Watermelon", "scientific": "Citrullus lanatus"},
            {"key": "banana", "name": "Banana", "scientific": "Musa acuminata"},
            {"key": "cocoa", "name": "Cocoa", "scientific": "Theobroma cacao"},
            {"key": "coffee", "name": "Coffee", "scientific": "Coffea arabica"},
            {"key": "strawberry", "name": "Strawberry", "scientific": "Fragaria x ananassa"},
            {"key": "plum", "name": "Plum", "scientific": "Prunus domestica"},
            {"key": "peach", "name": "Peach", "scientific": "Prunus persica"},
            {"key": "pear", "name": "Pear", "scientific": "Pyrus communis"},
        ],
        "vegetables": [
            {"key": "carrot", "name": "Carrot", "scientific": "Daucus carota"},
            {"key": "tomato", "name": "Tomato", "scientific": "Solanum lycopersicum"},
            {"key": "onion", "name": "Onion", "scientific": "Allium cepa"},
            {"key": "cucumber", "name": "Cucumber", "scientific": "Cucumis sativus"},
            {"key": "capsicum", "name": "Capsicum", "scientific": "Capsicum annuum"},
            {"key": "potato", "name": "Potato", "scientific": "Solanum tuberosum"},
        ],
    }


@router.post("/debug/grade", summary="Debug: grade an image directly (no segmentation)")
async def debug_grade(
    file: UploadFile = File(...),
    fruit_type: str = Query("potato"),
):
    import cv2
    import numpy as np
    from app.services.grading_service import grading_service
    from app.services.vegetable_grading_service import vegetable_grading_service

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    _VEG = {"carrot", "tomato", "onion", "cucumber", "capsicum", "potato"}
    if fruit_type.lower().strip() in _VEG:
        res = vegetable_grading_service.grade_vegetable(img, fruit_type, "debug")
    else:
        res = grading_service.grade_fruit(img, fruit_type, "debug")
    result = {
        "grade": res["grade"],
        "confidence": res["confidence"],
        "defect_score": res["defect_score"],
        "defects": res["defects"],
    }
    if "disease" in res:
        result["disease"] = res["disease"]
    return result


@router.get("/stats", summary="Dashboard aggregate statistics")
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Returns total scans, fruits analyzed, good rate, and estimated value from DB."""
    from sqlalchemy import func
    from app.database.models import AnalysisSession, BasketFruit

    total_scans_q = await db.execute(select(func.count(AnalysisSession.session_id)))
    total_scans = total_scans_q.scalar() or 0

    total_fruits_q = await db.execute(select(func.count(BasketFruit.id)))
    total_fruits = total_fruits_q.scalar() or 0

    good_better_q = await db.execute(
        select(func.count(BasketFruit.id)).where(
            BasketFruit.grade.in_(["Good", "Better"])
        )
    )
    good_better_count = good_better_q.scalar() or 0
    good_rate = round(good_better_count / total_fruits * 100) if total_fruits else 0

    BASE_PRICES = {"mango": 52.50, "grapes": 45.00, "pineapple": 85.00, "pomegranate": 40.00, "potato": 30.00, "watermelon": 40.00, "banana": 35.00, "cocoa": 50.00, "coffee": 60.00, "strawberry": 25.00, "plum": 30.00, "peach": 35.00, "pear": 30.00}
    QUALITY_MULTIPLIERS = {"Good": 1.0, "Better": 0.75, "Reject": 0.40}

    all_fruits_q = await db.execute(select(BasketFruit.fruit_type, BasketFruit.grade))
    total_value = 0.0
    for fruit_type, grade in all_fruits_q.all():
        base = BASE_PRICES.get((fruit_type or "").lower(), 50.0)
        mult = QUALITY_MULTIPLIERS.get(grade, 0.75)
        total_value += base * mult

    return {
        "total_scans": total_scans,
        "total_fruits": total_fruits,
        "good_rate": good_rate,
        "total_value": round(total_value, 2),
    }

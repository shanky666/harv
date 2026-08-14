"""
Mango-Only Prediction API Router
Provides endpoint POST /analyze-mango for end-to-end mango classification.
"""
import os
import uuid
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File
from loguru import logger
from app.core.config import settings
from app.services.quality_prediction import predict_quality

router = APIRouter(tags=["Mango Analysis"])

@router.post("/analyze-mango")
async def analyze_mango(file: UploadFile = File(...)):
    logger.info("POST /analyze-mango called")
    # 1. Save uploaded image to storage/uploads/original/
    safe_filename = "".join(c for c in os.path.basename(file.filename) if c.isalnum() or c in "._- ")
    original_filename = f"{uuid.uuid4()}_{safe_filename}"
    dest_dir = os.path.join(settings.STORAGE_BASE_PATH, "uploads", "original")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, original_filename)
    
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded image: {e}")
        raise HTTPException(status_code=500, detail="Could not save uploaded image.")

    # 2. Run reusable prediction function for mango
    try:
        result = predict_quality("mango", dest_path)
    except Exception as err:
        logger.error(f"Quality prediction failed: {err}")
        raise HTTPException(status_code=500, detail=f"Quality prediction failed: {err}")
        
    # Return response
    quality = result["quality"]
    confidence_pct = round(result["confidence"] * 100.0, 1)
    
    # Calculate shelf life and price dynamically based on quality rules
    from app.services.fruit_registry_service import fruit_registry
    from app.services.basket_analysis_service import parse_price
    
    fruit_config = fruit_registry.get_fruit("mango")
    if fruit_config:
        shelf_rules = fruit_config.get("shelf_life_rules", {})
        shelf_life_days = int(shelf_rules.get(quality, 5))
        
        market_rules = fruit_config.get("market_rules", {})
        rule_val = market_rules.get(quality, {})
        price_range = rule_val.get("price_range", "₹60/kg") if isinstance(rule_val, dict) else "₹60/kg"
        price_per_kg = parse_price(price_range)
    else:
        shelf_life_days = 5
        price_per_kg = 60.0
        
    market_price = round(price_per_kg * 0.3, 2)
    
    return {
        "fruit_type": "mango",
        "quality": quality,
        "confidence": confidence_pct,
        "shelf_life": f"{shelf_life_days} days",
        "market_price": market_price,
        "original_image_path": f"storage/uploads/original/{original_filename}"
    }




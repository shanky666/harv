import os
import pytest
import uuid
import tempfile
import cv2
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image

from app.services.classification_service import classification_service
from app.services.basket_analysis_service import basket_analysis_service
from app.database.models import AnalysisSession, BasketFruit

def test_classification_service_fallback():
    # 1. Non-existent path returns unknown class and 0.0 confidence (no hardcoded fallback)
    res = classification_service.classify_fruit("non_existent.jpg")
    assert "fruit_type" in res
    assert res["fruit_type"] == "unknown"
    assert res["confidence"] == 0.0

    # 2. Non-existent path with keyword in name also returns unknown (filename-based inference removed)
    res_orange = classification_service.classify_fruit("storage/uploads/orange_crop.jpg")
    assert res_orange["fruit_type"] == "unknown"
    assert res_orange["confidence"] == 0.0

    res_pomegranate = classification_service.classify_fruit("storage/uploads/pomegranate_crop.jpg")
    assert res_pomegranate["fruit_type"] == "unknown"
    assert res_pomegranate["confidence"] == 0.0


def test_classification_color_heuristic():
    temp_dir = tempfile.mkdtemp()
    temp_img_path = os.path.join(temp_dir, "crop.jpg")
    try:
        # Create a red crop (representing Pomegranate)
        # Red in BGR: (0, 0, 255)
        red_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        red_crop[:, :] = (0, 0, 255)
        cv2.imwrite(temp_img_path, red_crop)

        res = classification_service.classify_fruit(temp_img_path)
        assert res["fruit_type"] == "pomegranate"
        assert res["confidence"] > 0.5

        # Create a green crop (representing Grapes)
        # Green in BGR: (0, 255, 0)
        green_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        np.random.seed(42)
        g_channel = np.random.randint(20, 255, (100, 100), dtype=np.uint8)
        green_crop[:, :, 0] = 0 # Blue
        green_crop[:, :, 1] = g_channel # Green
        green_crop[:, :, 2] = 0 # Red
        cv2.imwrite(temp_img_path, green_crop)

        res = classification_service.classify_fruit(temp_img_path)
        assert res["fruit_type"] == "grapes"
        assert res["confidence"] > 0.5
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_basket_analysis_service_logic():
    db_mock = AsyncMock()
    session_id = uuid.uuid4()
    user_id = uuid.uuid4()
    
    yolo_detections = [
        {"fruit_id": "FRUIT_0001", "bbox": [10, 20, 100, 110], "confidence": 0.92, "fruit_type": "mango"}
    ]
    
    crop_detections = [
        {"fruit_id": "FRUIT_0001", "bbox": [10, 20, 100, 110], "confidence": 0.92, "fruit_type": "mango", "crop_path": "/storage/crop.jpg"}
    ]
    
    class_res = {"fruit_type": "mango", "confidence": 0.95}
    grade_res = {"grade": "Good", "confidence": 0.94}

    with patch("app.services.basket_analysis_service.yolo_service.detect_fruits", return_value=yolo_detections), \
         patch("app.services.basket_analysis_service.crop_service.crop_fruits", return_value=crop_detections), \
         patch("app.services.basket_analysis_service.classification_service.classify_fruit", return_value=class_res), \
         patch("app.services.basket_analysis_service.grading_service.grade_fruit", return_value=grade_res), \
         patch("os.path.exists", return_value=True):
             
        res = await basket_analysis_service.analyze_basket(db_mock, "/storage/basket.jpg", user_id)
        
        assert res["total_fruits"] == 1
        assert res["fruits"][0]["fruit_type"] == "mango"
        assert res["fruits"][0]["grade"] == "Good"
        assert res["summary"]["good"] == 1
        
        # Verify db persistence calls
        assert db_mock.add.called
        assert db_mock.commit.called

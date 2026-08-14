import os
import sys
import pytest
import numpy as np

# Add backend to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services.grading_service import GradingService

def test_grading_service_cache_resolution():
    service = GradingService()
    service.load_model()
    
    # Try resolving model for mango
    mango_model = service.get_model_for_fruit("mango")
    assert mango_model is not None
    
    # Try resolving model for orange
    orange_model = service.get_model_for_fruit("orange")
    assert orange_model is not None

def test_grading_service_inference(tmp_path):
    service = GradingService()
    
    # Create a dummy crop image
    crop_path = os.path.join(tmp_path, "crop_001.jpg")
    import cv2
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(crop_path, dummy_img)
    
    # Run grade fruit
    grade_info = service.grade_fruit(crop_path, fruit_type="mango")
    
    assert "grade" in grade_info
    assert "confidence" in grade_info
    assert "defect_score" in grade_info
    assert "defects" in grade_info
    assert grade_info["grade"] in ["Good", "Better", "Medium", "Reject"]
    assert 0.0 <= grade_info["confidence"] <= 1.0
    assert 0.0 <= grade_info["defect_score"] <= 1.0
    assert isinstance(grade_info["defects"], list)

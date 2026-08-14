import os
import tempfile
import uuid
import pytest
import numpy as np
import cv2
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from app.utils.image_utils import save_image
from app.services.grading_service import grading_service, GRADES
from app.database.crud import update_fruit_grade
from app.database.models import DetectedFruit
from app.api.grading import grade_fruits


@pytest.fixture
def dummy_crop_path():
    """Generates a temporary dummy crop image for testing."""
    fd, temp_path = tempfile.mkstemp(suffix=".jpg")
    try:
        # Create a simple green image (simulating a good fruit)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = (0, 180, 0)  # Green color
        # Draw a dark circle to simulate a small defect spot
        cv2.circle(img, (50, 50), 5, (20, 20, 20), -1)
        cv2.imwrite(temp_path, img)
        os.close(fd)
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_preprocess_image(dummy_crop_path):
    """Verifies that preprocessing loads, resizes, and normalizes the crop correctly."""
    processed = grading_service.preprocess_image(dummy_crop_path)
    
    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.float32
    assert np.max(processed) <= 1.0
    assert np.min(processed) >= 0.0


def test_predict_grade(dummy_crop_path):
    """Verifies predict_grade output structure (grade, confidence, defect_score, defects)."""
    grade, confidence, defect_score, defects = grading_service.predict_grade(dummy_crop_path)
    
    assert grade in GRADES
    assert 0.0 <= confidence <= 1.0
    assert 0.0 <= defect_score <= 1.0
    assert isinstance(defects, list)


def test_mock_grading(dummy_crop_path):
    """Tests that grade_all_fruits processes multiple crops correctly."""
    fruits = [
        {"fruit_id": "FRUIT_0001", "crop_path": dummy_crop_path, "fruit_type": "mango"},
        {"fruit_id": "FRUIT_0002", "crop_path": dummy_crop_path, "fruit_type": "orange"}
    ]
    
    results = grading_service.grade_all_fruits(fruits)
    assert len(results) == 2
    
    for res in results:
        assert res["fruit_id"] in ["FRUIT_0001", "FRUIT_0002"]
        assert res["grade"] in GRADES
        assert "confidence" in res
        assert "defect_score" in res
        assert "defects" in res


def test_statistics_generation():
    """Tests that aggregate grade statistics counts are generated correctly."""
    graded_fruits = [
        {"grade": "Good"},
        {"grade": "Better"},
        {"grade": "Good"},
        {"grade": "Medium"},
        {"grade": "Reject"}
    ]
    
    stats = grading_service.generate_grade_statistics(graded_fruits)
    assert stats["good"] == 2
    assert stats["better"] == 1
    assert stats["medium"] == 1
    assert stats["reject"] == 1


@pytest.mark.asyncio
async def test_database_persistence():
    """Mocks the DB session and verifies that update_fruit_grade writes all new columns."""
    db_mock = AsyncMock()
    fruit_id = str(uuid.uuid4())
    
    # Mock get_fruit_by_id in crud
    mock_fruit = MagicMock()
    mock_fruit.fruit_id = uuid.UUID(fruit_id)
    
    with patch("app.database.crud.get_fruit_by_id", return_value=mock_fruit):
        await update_fruit_grade(
            db=db_mock,
            fruit_id=fruit_id,
            grade="Better",
            grade_confidence=0.92,
            defect_score=0.08,
            shelf_life="5-7 days",
            market="Metro Mandi",
            predicted_at=datetime.utcnow()
        )
        
    # Check that execute was called to update the record
    assert db_mock.execute.called
    # Check that commit was called
    assert db_mock.commit.called


@pytest.mark.asyncio
async def test_api_response_validation(dummy_crop_path):
    """Tests the /scan/grade/{scan_id} endpoint by mocking dependencies and verifying JSON format."""
    db_mock = AsyncMock()
    scan_id = uuid.uuid4()
    
    # Mock data
    mock_scan = MagicMock()
    mock_scan.scan_id = scan_id
    
    # Two fruits detected earlier
    mock_fruit1 = MagicMock(spec=DetectedFruit)
    mock_fruit1.fruit_id = uuid.uuid4()
    mock_fruit1.fruit_type = "mango"
    mock_fruit1.crop_path = f"storage/crops/{scan_id}/FRUIT_0001.jpg"
    
    mock_fruit2 = MagicMock(spec=DetectedFruit)
    mock_fruit2.fruit_id = uuid.uuid4()
    mock_fruit2.fruit_type = "orange"
    mock_fruit2.crop_path = f"storage/crops/{scan_id}/FRUIT_0002.jpg"
    
    fruits_list = [mock_fruit1, mock_fruit2]
    
    # Mock the CRUD calls
    with patch("app.api.grading.get_scan", return_value=mock_scan), \
         patch("app.api.grading.get_fruits_by_scan", return_value=fruits_list), \
         patch("app.api.grading.update_fruit_grade", return_value=None), \
         patch("os.path.exists", return_value=True):
             
        # Call the endpoint handler directly
        response = await grade_fruits(
            scan_id=scan_id,
            db=db_mock,
            current_user=MagicMock()
        )
        
        # Verify response matches GradeCountResponse structure
        assert response.scan_id == str(scan_id)
        assert response.total_fruits == 2
        assert hasattr(response, "good")
        assert hasattr(response, "better")
        assert hasattr(response, "medium")
        assert hasattr(response, "reject")
        assert len(response.fruits) == 2
        
        # Check individual fruit details
        fruit_item = response.fruits[0]
        assert fruit_item.fruit_id in ["FRUIT_0001", "FRUIT_0002"]
        assert fruit_item.grade in GRADES
        assert 0.0 <= fruit_item.confidence <= 1.0
        assert isinstance(fruit_item.defects, list)

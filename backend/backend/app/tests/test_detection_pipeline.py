import os
import tempfile
import uuid
import pytest
import numpy as np
import cv2
from app.utils.image_utils import clip_bbox, crop_roi, load_image, save_image
from app.services.yolo_service import yolo_service
from app.services.crop_service import crop_service
from app.core.config import settings


@pytest.fixture
def mock_image_path():
    """Creates a temporary dummy image file for testing."""
    fd, temp_path = tempfile.mkstemp(suffix=".jpg")
    try:
        # Create a black image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw some rectangles to simulate objects
        cv2.rectangle(img, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.rectangle(img, (300, 150), (450, 350), (255, 255, 255), -1)
        cv2.imwrite(temp_path, img)
        os.close(fd)
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_clip_bbox():
    """Verifies that bounding boxes are clamped correctly inside image limits."""
    img_shape = (480, 640, 3)
    
    # Fully inside
    assert clip_bbox([100, 100, 200, 200], img_shape) == [100, 100, 200, 200]
    
    # Exceeding bounds
    assert clip_bbox([-50, -50, 700, 500], img_shape) == [0, 0, 640, 480]
    
    # Inverted coordinates
    clipped = clip_bbox([200, 200, 100, 100], img_shape)
    assert clipped[0] < clipped[2]
    assert clipped[1] < clipped[3]


def test_yolo_id_generation():
    """Verifies fruit ID naming convention (FRUIT_0001)."""
    assert yolo_service.generate_fruit_id(1) == "FRUIT_0001"
    assert yolo_service.generate_fruit_id(15) == "FRUIT_0015"
    assert yolo_service.generate_fruit_id(9999) == "FRUIT_9999"


def test_mock_detect_and_response(mock_image_path):
    """Tests that the yolo service successfully runs mock detection and structures output JSON."""
    scan_id = str(uuid.uuid4())
    
    from unittest.mock import patch
    with patch("app.services.classification_service.classification_service.classify_fruit", 
               return_value={"fruit_type": "mango", "confidence": 0.95}):
        # Detect
        detections = yolo_service.detect_fruits(mock_image_path, scan_id)
        assert len(detections) >= 1
        
        # Check item keys
        for det in detections:
            assert "fruit_id" in det
            assert det["fruit_id"].startswith("FRUIT_")
            assert "bbox" in det
            assert len(det["bbox"]) == 4
            assert "confidence" in det
            assert 0.60 <= det["confidence"] <= 1.0
            
        # Test response formatting
        response = yolo_service.create_detection_response(scan_id, detections)
        assert response["scan_id"] == scan_id
        assert response["total_fruits"] == len(detections)
        
        first_det = response["detections"][0]
        assert first_det["fruit_id"].startswith("FRUIT_")
        assert len(first_det["bbox"]) == 4
        assert first_det["crop_path"] == f"storage/crops/{scan_id}/{first_det['fruit_id']}.jpg"


def test_crop_service(mock_image_path):
    """Tests that the crop service correctly extracts and saves ROIs to disk."""
    scan_id = str(uuid.uuid4())
    detections = [
        {"fruit_id": "FRUIT_0001", "fruit_type": "mango", "bbox": [100, 100, 200, 200], "confidence": 0.9},
        {"fruit_id": "FRUIT_0002", "fruit_type": "orange", "bbox": [300, 150, 450, 350], "confidence": 0.85}
    ]
    
    # Crop
    updated_dets = crop_service.crop_fruits(mock_image_path, detections, scan_id)
    
    assert len(updated_dets) == 2
    for det in updated_dets:
        assert "crop_path" in det
        assert os.path.exists(det["crop_path"])
        
        # Verify the saved crop image can be loaded and is not empty
        crop_img = load_image(det["crop_path"])
        assert crop_img.shape[0] > 0
        assert crop_img.shape[1] > 0
        
        # Cleanup file
        os.remove(det["crop_path"])
        
    # Cleanup temporary directory created for the scan
    scan_dir = os.path.dirname(updated_dets[0]["crop_path"])
    if os.path.exists(scan_dir):
        os.rmdir(scan_dir)

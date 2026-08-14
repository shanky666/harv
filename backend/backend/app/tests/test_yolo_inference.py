import os
import sys
import pytest
import numpy as np
import cv2

# Add backend to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services.yolo_service import YOLODetectionService
from app.core.config import settings

def test_yolo_service_model_loading():
    # Instantiate service
    service = YOLODetectionService()
    service.load_model()
    # It should successfully initialize (either real model or fallback to None which enables mock mode)
    # The load method itself must not raise exceptions.
    assert hasattr(service, "model")

def test_yolo_service_inference(tmp_path):
    service = YOLODetectionService()
    service.load_model()
    
    # Create a dummy image with a simulated red fruit
    img_path = os.path.join(tmp_path, "dummy.jpg")
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(dummy_img, (320, 240), 80, (0, 0, 255), -1)
    cv2.imwrite(img_path, dummy_img)
    
    # Run fruit detection
    detections = service.detect_fruits(img_path, scan_id="test_scan_001")
    
    assert isinstance(detections, list)
    assert len(detections) > 0
    for det in detections:
        assert "fruit_id" in det
        assert "fruit_type" in det
        assert "bbox" in det
        assert "confidence" in det
        assert det["fruit_id"].startswith("FRUIT_")
        assert len(det["bbox"]) == 4

def test_yolo_service_black_image(tmp_path):
    service = YOLODetectionService()
    service.load_model()
    
    # Create a completely black image
    img_path = os.path.join(tmp_path, "black.jpg")
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(img_path, dummy_img)
    
    # Run fruit detection
    detections = service.detect_fruits(img_path, scan_id="test_scan_black")
    
    # Should detect 0 fruits
    assert isinstance(detections, list)
    assert len(detections) == 0

def test_yolo_response_formatting():
    service = YOLODetectionService()
    mock_detections = [
        {"fruit_id": "FRUIT_0001", "bbox": [10, 20, 100, 200], "confidence": 0.95, "crop_path": None}
    ]
    resp = service.create_detection_response("test_scan_123", mock_detections)
    assert resp["scan_id"] == "test_scan_123"
    assert resp["total_fruits"] == 1
    assert len(resp["detections"]) == 1
    assert resp["detections"][0]["fruit_id"] == "FRUIT_0001"
    assert resp["detections"][0]["crop_path"] == "storage/crops/test_scan_123/FRUIT_0001.jpg"

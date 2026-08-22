"""
Unit test for Fruit Type Verification & Mismatch Detection
Verifies that selecting Grapes and uploading a Pineapple crop flags a mismatch.
"""
import pytest
import numpy as np
import cv2

from app.services.classification_service import classification_service
from app.services.grading_service import grading_service


def test_grapes_vs_pineapple_mismatch_verification():
    """Simulates a golden-brown textured crop (Pineapple) verified against expected 'grapes'."""
    # Create a 224x224 crop with golden-brown hue (Pineapple profile) and high texture variance
    crop_pineapple = np.zeros((224, 224, 3), dtype=np.uint8)
    # Fill with golden yellow/brown HSV (B=30, G=120, R=180)
    crop_pineapple[:, :] = (30, 120, 180)
    # Add texture noise to simulate skin texture
    noise = np.random.randint(-40, 40, (224, 224, 3), dtype=np.int16)
    crop_pineapple = np.clip(crop_pineapple.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 1. Test Classification Verification directly
    verification = classification_service.verify_fruit(crop_pineapple, expected_fruit="grapes")
    assert verification["is_match"] is False
    assert "Image Mismatch" in verification["warning"]

    # 2. Test Grading Service response for mismatched fruit
    grade_res = grading_service.grade_fruit(crop_pineapple, fruit_type="grapes")
    assert grade_res["grade"] == "Mismatch"
    assert "fruit_type_mismatch" in grade_res["defects"]
    assert "warning" in grade_res


def test_auto_classification_mode():
    """Tests that expected_fruit='auto' automatically detects fruit type without flagging mismatch."""
    crop_strawberry = np.zeros((224, 224, 3), dtype=np.uint8)
    crop_strawberry[:, :] = (30, 30, 220)  # Vibrant red BGR

    verification = classification_service.verify_fruit(crop_strawberry, expected_fruit="auto")
    assert verification["is_match"] is True
    assert verification["predicted_fruit"] == "strawberry"


def test_strawberry_vs_mango_mismatch_verification():
    """Tests vibrant red crop (Strawberry) against expected 'mango'."""
    crop_strawberry = np.zeros((224, 224, 3), dtype=np.uint8)
    crop_strawberry[:, :] = (20, 20, 210)  # Bright red

    verification = classification_service.verify_fruit(crop_strawberry, expected_fruit="mango")
    assert verification["is_match"] is False
    assert "Image Mismatch" in verification["warning"]


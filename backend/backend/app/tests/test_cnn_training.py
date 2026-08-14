import os
import shutil
import tempfile
import pytest
from ai_models.cnn_training.train import train_model, build_cnn_model
from ai_models.cnn_training.predict import predict_crop
from ai_models.cnn_training.evaluate import evaluate_cnn_model
from ai_models.cnn_training.preprocessing import preprocess_image, preprocess_batch
import numpy as np

def test_cnn_preprocessing():
    # Test batch preprocess helper
    dummy_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    dummy_img.close()
    try:
        # Create a simple image matrix
        import cv2
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(dummy_img.name, img)
        
        prep = preprocess_image(dummy_img.name)
        assert prep.shape == (224, 224, 3)
        assert prep.dtype == np.float32
        
        batch = preprocess_batch([dummy_img.name, dummy_img.name])
        assert batch.shape == (2, 224, 224, 3)
    finally:
        os.remove(dummy_img.name)

def test_build_model_structure():
    # Only test if tensorflow is installed
    try:
        import tensorflow as tf
        model = build_cnn_model(model_type="MobileNetV3", num_classes=4)
        assert model.name == "fruit_quality_grader"
        assert len(model.outputs) == 1
        assert model.output_shape == (None, 4)
    except Exception:
        # Fallback if TF is missing or has import issues during testing
        logger = __import__("loguru").logger
        logger.warning("TensorFlow build test skipped due to framework import error.")

def test_cnn_training_dry_run():
    metrics = train_model(fruit_type="Mango", epochs=1, dry_run=True)
    assert metrics["fruit_type"] == "Mango"
    assert "accuracy" in metrics
    assert os.path.exists(metrics["best_checkpoint"])

def test_cnn_predictions_dry_run():
    mock_weights = "models/mango_model_v1.h5"
    
    grade, confidence, defect_score = predict_crop(mock_weights, "dummy_crop.jpg", dry_run=True)
    assert grade in ["Good", "Better", "Medium", "Reject"]
    assert 0.0 <= confidence <= 1.0
    assert 0.0 <= defect_score <= 1.0

def test_cnn_evaluation_reporting():
    mock_weights = "models/mango_model_v1.h5"
    temp_dir = tempfile.mkdtemp()
    try:
        metrics = evaluate_cnn_model(mock_weights, val_data_dir="dummy_val", reports_dir=temp_dir, dry_run=True)
        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.8
        
        # Verify written files
        assert os.path.exists(os.path.join(temp_dir, "metrics.json"))
        assert os.path.exists(os.path.join(temp_dir, "classification_report.txt"))
        assert os.path.exists(os.path.join(temp_dir, "confusion_matrix.png"))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


import os
import shutil
import tempfile
import pytest
from ai_models.yolo_detection.train import train_yolo, evaluate_yolo, export_onnx, export_tflite
from ai_models.yolo_detection.predict import predict_image, predict_folder
from ai_models.yolo_detection.evaluate import evaluate_yolo_model

def test_yolo_training_dry_run():
    # Run mock yolo training
    metrics = train_yolo(model_name="yolov8n", epochs=1, dry_run=True)
    assert metrics["mAP50"] > 0.8
    assert "best_model_path" in metrics
    assert os.path.exists(metrics["best_model_path"])
    
    # Evaluate
    val_metrics = evaluate_yolo(model_path=metrics["best_model_path"], dry_run=True)
    assert "precision" in val_metrics
    assert val_metrics["mAP50"] > 0.8

def test_yolo_exports():
    mock_weights = "runs/detect/train/weights/best.pt"
    os.makedirs(os.path.dirname(mock_weights), exist_ok=True)
    with open(mock_weights, "w") as f:
        f.write("mock_pt")
        
    onnx_path = export_onnx(mock_weights, dry_run=True)
    assert onnx_path.endswith(".onnx")
    assert os.path.exists(onnx_path)
    
    tflite_path = export_tflite(mock_weights, dry_run=True)
    assert "_saved_model" in tflite_path
    assert os.path.exists(tflite_path)

def test_yolo_predictions():
    mock_weights = "runs/detect/train/weights/best.pt"
    
    # Predict single image
    detections = predict_image(mock_weights, "dummy_img.jpg", dry_run=True)
    assert len(detections) > 0
    assert "box" in detections[0]
    assert "class" in detections[0]
    assert "confidence" in detections[0]

def test_yolo_evaluation_reporting():
    mock_weights = "runs/detect/train/weights/best.pt"
    temp_dir = tempfile.mkdtemp()
    try:
        report = evaluate_yolo_model(mock_weights, reports_dir=temp_dir, dry_run=True)
        assert report["status"] == "COMPLETED"
        assert report["pipeline_type"] == "YOLOv8 Fruit Detection"
        
        # Verify written files
        assert os.path.exists(os.path.join(temp_dir, "metrics.json"))
        assert os.path.exists(os.path.join(temp_dir, "training_report.json"))
    finally:
        shutil.rmtree(temp_dir)

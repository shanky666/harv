import os
import json
from loguru import logger

def verify_all_models():
    logger.info("=========================================")
    logger.info("         VERIFYING ALL MODELS            ")
    logger.info("=========================================")
    
    # Resolve paths
    api_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(api_dir, "..", ".."))
    models_dir = os.path.join(workspace_root, "models")
    
    # List of models to verify
    models_to_check = [
        {"name": "yolo", "filename": "yolov8n.pt", "type": "yolo"},
        {"name": "basket_classifier", "filename": "basket_classifier.h5", "type": "keras", "classes": ["grapes", "mango", "pineapple", "pomegranate"]},
        {"name": "classifier", "filename": "fruit_classifier.h5", "type": "keras", "classes": ["grapes", "mango", "pineapple", "pomegranate"]},
        {"name": "mango_quality", "filename": "mango_quality_ft.h5", "type": "keras", "classes": ["Better", "Good", "Medium", "Reject"]},
        {"name": "grapes_quality", "filename": "grapes_quality_ft.h5", "type": "keras", "classes": ["Better", "Good", "Medium", "Reject"]},
        {"name": "pineapple_quality", "filename": "pineapple_quality_ft.h5", "type": "keras", "classes": ["Better", "Good", "Medium", "Reject"]},
        {"name": "pomegranate_quality", "filename": "pomegranate_quality_ft.h5", "type": "keras", "classes": ["Better", "Good", "Medium", "Reject"]},
    ]
    
    for m in models_to_check:
        path = os.path.join(models_dir, m["filename"])
        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            print(f"Model Path: {path} (NOT FOUND)")
            continue
            
        file_size = os.path.getsize(path)
        size_mb = round(file_size / (1024 * 1024), 4)
        
        # Check if placeholder
        is_placeholder = file_size < 50 * 1024
        if not is_placeholder:
            try:
                with open(path, "r", errors="ignore") as f:
                    content = f.read(100)
                    if any(x in content for x in ["MOCK_WEIGHTS", "MOCK_YOLO", "MOCK_CNN"]):
                        is_placeholder = True
            except Exception:
                pass
                
        if is_placeholder:
            logger.warning(f"[WARNING] Model file {path} is a placeholder. Using fallback heuristic.")
            print(f"Model Path: {path}")
            print(f"  File Size: {size_mb} MB (Placeholder)")
            print(f"  Classes: {m.get('classes', 'N/A')}")
            print(f"  Input Shape: N/A")
            print(f"  Output Shape: N/A")
            print(f"  [STATUS] DISABLED/HEURISTIC FALLBACK")
            continue
            
        # Try loading real model
        try:
            if m["type"] == "yolo":
                from ultralytics import YOLO
                yolo = YOLO(path)
                classes = list(yolo.names.values())
                input_shape = "[1, 3, 640, 640]"
                output_shape = "[1, 84, 8400]"
            else:
                import keras
                # Mute tensorflow logs during verification
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                keras_model = keras.models.load_model(path, compile=False)
                classes = m.get("classes", [])
                input_shape = str(keras_model.input_shape)
                output_shape = str(keras_model.output_shape)
                
            print(f"Model Path: {path}")
            print(f"  File Size: {size_mb} MB")
            print(f"  Classes: {classes}")
            print(f"  Input Shape: {input_shape}")
            print(f"  Output Shape: {output_shape}")
            print(f"  [STATUS] LOADED SUCCESS")
        except Exception as e:
            logger.error(f"Model file {path} failed to load (corrupted): {e}")
            print(f"Model Path: {path}")
            print(f"  File Size: {size_mb} MB")
            print(f"  Classes: {m.get('classes', 'N/A')}")
            print(f"  [STATUS] CORRUPTED/DISABLED")

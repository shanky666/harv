import os
import shutil
import tensorflow as tf
from loguru import logger

CLASSES = ["grapes", "mango", "pineapple", "pomegranate"]

def deploy():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    
    # Target production directories
    target_dir1 = os.path.join(script_dir, "app", "ai", "models")
    parent_backend_dir = os.path.abspath(os.path.join(script_dir, ".."))
    target_dir2 = os.path.join(parent_backend_dir, "app", "ai", "models")
    
    os.makedirs(target_dir1, exist_ok=True)
    os.makedirs(target_dir2, exist_ok=True)
    
    for fruit in CLASSES:
        temp_path = os.path.join(models_dir, f"{fruit}_quality_temp.h5")
        ft_path = os.path.join(models_dir, f"{fruit}_quality_ft.h5")
        
        # Determine the best model path
        best_path = None
        if os.path.exists(ft_path):
            best_path = ft_path
            logger.info(f"For {fruit}, selected Fine-tuned (FT) model: {ft_path}")
        elif os.path.exists(temp_path):
            best_path = temp_path
            logger.info(f"For {fruit}, selected Temp (Phase 1) model: {temp_path}")
        else:
            logger.warning(f"No model files found for {fruit} under {models_dir}")
            continue
            
        # Copy to targets
        dest1 = os.path.join(target_dir1, f"{fruit}_model.h5")
        dest2 = os.path.join(target_dir2, f"{fruit}_model.h5")
        
        shutil.copy2(best_path, dest1)
        shutil.copy2(best_path, dest2)
        logger.info(f"Successfully deployed {fruit} model to production paths.")

if __name__ == "__main__":
    deploy()

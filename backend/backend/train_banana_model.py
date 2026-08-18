import os
import sys
from loguru import logger
from train_grading_model import train_pipeline_for_fruit

if __name__ == "__main__":
    epochs = 5
    if len(sys.argv) > 1:
        epochs = int(sys.argv[1])
    
    logger.info(f"Starting banana quality grading model training for {epochs} epochs...")
    try:
        train_pipeline_for_fruit("banana", epochs=epochs)
        logger.info("Banana model training finished successfully.")
    except Exception as e:
        logger.error(f"Error during banana model training: {e}")
        sys.exit(1)

import os
import uuid
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Any
from loguru import logger

def load_image(path: str) -> np.ndarray:
    """
    Safely reads an image from the given path using OpenCV.
    Performs validation checks to prevent loading failures.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image path does not exist: {path}")
    
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot load image (invalid image format or corrupt file): {path}")
    
    return img

def clip_bbox(bbox: List[int], img_shape: Tuple[int, ...]) -> List[int]:
    """
    Clips bounding box coordinates to ensure they fall within the image boundary limits.
    Prevents empty or out-of-bounds crops.
    """
    h, w = img_shape[:2]
    x1, y1, x2, y2 = bbox
    
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    # Ensure coordinates are ordered correctly
    if x1 >= x2:
        x2 = x1 + 1
    if y1 >= y2:
        y2 = y1 + 1
        
    return [x1, y1, x2, y2]

def crop_roi(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """
    Extracts a Region of Interest (ROI) crop from the image.
    Applies bounding box clipping automatically.
    """
    x1, y1, x2, y2 = clip_bbox([x1, y1, x2, y2], image.shape)
    return image[y1:y2, x1:x2]

def save_image(image: np.ndarray, path: str) -> str:
    """
    Saves an image to the specified path, creating directories if needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    success = cv2.imwrite(path, image)
    if not success:
        raise IOError(f"Failed to save image to path: {path}")
    return path

def resize_image(image: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Resizes the image to the target size.
    """
    return cv2.resize(image, size)

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizes pixel values of the image from [0, 255] to [0.0, 1.0].
    """
    return image.astype(np.float32) / 255.0

def generate_unique_filename(extension: str = "jpg") -> str:
    """
    Generates a unique filename using a UUID.
    """
    return f"{uuid.uuid4().hex}.{extension}"

def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """
    Draws bounding boxes, fruit class names, and confidence scores on the image.
    Supports drawing full quality grade labels.
    """
    annotated = image.copy()
    type_counts = {}
    
    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue
            
        x1, y1, x2, y2 = map(int, bbox)
        
        fruit_type = det.get("fruit_type", "unknown")
        confidence = det.get("confidence", 0.0)
        grade = det.get("grade")
        
        # Get or compute fruit index
        fruit_index = det.get("fruit_index")
        if fruit_index is None:
            type_counts[fruit_type] = type_counts.get(fruit_type, 0) + 1
            fruit_index = type_counts[fruit_type]
            
        conf_percent = int(confidence * 100)
        
        if grade:
            type_display = fruit_type.capitalize()
            label = f"{type_display} #{fruit_index} | {conf_percent}% | {grade}"
        else:
            label = f"{fruit_type} #{fruit_index} | {conf_percent}%"
            
        color = (0, 255, 0) # Green (BGR)
        if grade:
            grade_lower = grade.lower()
            if grade_lower in ["good"]:
                color = (0, 255, 0) # Green
            elif grade_lower in ["better", "excellent", "medium"]:
                color = (0, 165, 255) # Orange (BGR)
            else:
                color = (0, 0, 255) # Red
                
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = max(y1 - 10, 15)
        
        cv2.rectangle(annotated, (x1, text_y - text_h - 2), (x1 + text_w + 4, text_y + baseline), color, cv2.FILLED)
        cv2.putText(
            annotated, 
            label, 
            (x1 + 2, text_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1,
            lineType=cv2.LINE_AA
        )
    return annotated

def resize_image_to_limit(path: str, max_size: int = 1280):
    """
    Resizes the image at the given path in-place if any of its dimensions exceeds max_size.
    Preserves aspect ratio.
    """
    try:
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            if max(h, w) > max_size:
                if w > h:
                    new_w = max_size
                    new_h = int(h * (max_size / w))
                else:
                    new_h = max_size
                    new_w = int(w * (max_size / h))
                img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(path, img_resized)
                logger.info(f"Resized image {path} from {w}x{h} to {new_w}x{new_h}")
    except Exception as e:
        logger.error(f"Failed to resize image at {path}: {e}")

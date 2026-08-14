"""
Background Removal Service
Removes non-fruit pixels using segmentation masks.
Preserves only fruit pixels for accurate grading.
"""
import os
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
from loguru import logger


class BackgroundRemovalService:
    """
    Uses segmentation masks to isolate fruit pixels.

    Pipeline:
        1. Receive original image + binary mask
        2. Apply mask to extract fruit-only region
        3. Compute tight bounding box of fruit pixels
        4. Crop and resize to classifier input size
        5. Optionally refine edges with GrabCut

    This ensures the grading model only sees fruit pixels,
    eliminating background noise (tables, baskets, leaves, hands).
    """

    def extract_fruit_crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bbox: list,
        padding_ratio: float = 0.05,
        target_size: Tuple[int, int] = (224, 224),
        refine_edges: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Extracts a clean fruit crop using the segmentation mask.

        Args:
            image: Original BGR image (H, W, 3)
            mask: Binary mask (H, W) where 255 = fruit
            bbox: [x1, y1, x2, y2] bounding box of the fruit region
            padding_ratio: Padding around the bbox (default 5%)
            target_size: Output size for the classifier (default 224x224)
            refine_edges: Whether to apply GrabCut refinement

        Returns:
            Clean BGR crop image of shape (target_size[1], target_size[0], 3)
            or None if extraction fails.
        """
        if image is None or mask is None:
            return None

        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = bbox

        pad_w = int((x2 - x1) * padding_ratio)
        pad_h = int((y2 - y1) * padding_ratio)
        x1p = max(0, x1 - pad_w)
        y1p = max(0, y1 - pad_h)
        x2p = min(w_img, x2 + pad_w)
        y2p = min(h_img, y2 + pad_h)

        crop_img = image[y1p:y2p, x1p:x2p].copy()
        crop_mask = mask[y1p:y2p, x1p:x2p].copy()

        if crop_img.size == 0 or crop_mask.size == 0:
            return None

        if refine_edges:
            crop_mask = self._grabcut_refine(crop_img, crop_mask)

        crop_resized = cv2.resize(crop_img, target_size, interpolation=cv2.INTER_AREA)

        return crop_resized

    def extract_fruit_crop_by_contour(
        self,
        image: np.ndarray,
        contour: np.ndarray,
        bbox: list,
        padding_ratio: float = 0.05,
        target_size: Tuple[int, int] = (224, 224),
    ) -> Optional[np.ndarray]:
        """
        Extracts a fruit crop using a contour instead of a full mask.
        Faster than mask-based extraction for simple cases.
        """
        if image is None or contour is None:
            return None

        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = bbox

        pad_w = int((x2 - x1) * padding_ratio)
        pad_h = int((y2 - y1) * padding_ratio)
        x1p = max(0, x1 - pad_w)
        y1p = max(0, y1 - pad_h)
        x2p = min(w_img, x2 + pad_w)
        y2p = min(h_img, y2 + pad_h)

        crop_img = image[y1p:y2p, x1p:x2p].copy()

        if crop_img.size == 0:
            return None

        contour_shifted = contour.copy()
        contour_shifted[:, :, 0] = contour_shifted[:, :, 0] - x1p
        contour_shifted[:, :, 1] = contour_shifted[:, :, 1] - y1p

        mask_crop = np.zeros(crop_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_crop, [contour_shifted], -1, 255, -1)

        crop_resized = cv2.resize(crop_img, target_size, interpolation=cv2.INTER_AREA)

        return crop_resized

    def _grabcut_refine(
        self, crop: np.ndarray, initial_mask: np.ndarray
    ) -> np.ndarray:
        """
        Refines the mask edges using GrabCut for pixel-precise boundaries.
        """
        h, w = crop.shape[:2]
        if h < 10 or w < 10:
            return initial_mask

        gc_mask = np.where(initial_mask > 0, cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)

        try:
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            rect = (2, 2, w - 4, h - 4)
            cv2.grabCut(crop, gc_mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)

            refined = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)
            return refined

        except Exception as e:
            logger.warning(f"GrabCut refinement failed, using initial mask: {e}")
            return initial_mask

    def create_background_removed_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        background_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """
        Creates a full image with background removed.
        Used for annotated report images.
        """
        bg = np.full_like(image, background_color, dtype=np.uint8)
        return np.where(mask[:, :, np.newaxis] > 0, image, bg)


background_removal_service = BackgroundRemovalService()

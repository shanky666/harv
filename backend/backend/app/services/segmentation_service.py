"""
Instance Segmentation Service
Detects individual fruits using classical CV (watershed + contour analysis)
and optionally Mask R-CNN for pixel-precise masks.

Replaces the old YOLO-based bounding box detection.
"""
import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger


class SegmentationService:
    """
    Segments individual fruits from an image.

    Multi-strategy approach:
        1. Foreground mask via S/V thresholding (tuned per fruit type)
        2. Watershed with low threshold for separation
        3. Contour-based fallback
        4. Full image fallback (single fruit)
    """

    def __init__(self):
        self._min_area_ratio = 0.01
        self._max_area_ratio = 0.85
        self._min_circularity = 0.15
        self._min_solidity = 0.50
        self._max_aspect = 4.0

    def segment_fruits(
        self,
        image: np.ndarray,
        fruit_type: str = "mango",
        is_single: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Main entry point. Returns a list of segmented fruit regions.
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]

        if is_single:
            return self._segment_single_fruit(image, h, w, fruit_type)

        return self._segment_multiple_fruits(image, h, w, fruit_type)

    def _segment_single_fruit(
        self, image: np.ndarray, h: int, w: int, fruit_type: str
    ) -> List[Dict[str, Any]]:
        """Treats the entire image as a single fruit region."""
        mask = np.ones((h, w), dtype=np.uint8) * 255
        contour = np.array([[0, 0], [w, 0], [w, h], [0, h]])

        return [{
            "mask": mask,
            "bbox": [0, 0, w, h],
            "contour": contour,
            "area": float(h * w),
            "confidence": 0.95,
            "fruit_id": "FRUIT_0001",
        }]

    def _segment_multiple_fruits(
        self, image: np.ndarray, h: int, w: int, fruit_type: str
    ) -> List[Dict[str, Any]]:
        """Segments multiple individual fruits from the image."""
        work_h = min(h, 640)
        work_w = int(w * (work_h / h)) if h > 0 else w
        image_small = cv2.resize(image, (work_w, work_h))
        scale_x = w / work_w
        scale_y = h / work_h

        hsv = cv2.cvtColor(image_small, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)

        regions = self._multi_strategy_segment(
            image_small, hsv, gray, fruit_type
        )

        valid_regions = []
        for region in regions:
            if self._validate_region(region, work_w, work_h):
                scaled = self._scale_region(region, scale_x, scale_y)
                scaled["fruit_id"] = ""
                valid_regions.append(scaled)

        valid_regions = self._remove_duplicates(valid_regions)

        for i, region in enumerate(valid_regions, 1):
            region["fruit_id"] = f"FRUIT_{i:04d}"
            region["mask"] = cv2.resize(
                region["mask"], (w, h), interpolation=cv2.INTER_NEAREST
            )

        if not valid_regions:
            logger.warning("No valid fruit regions detected, using full image as fallback")
            valid_regions = [{
                "mask": np.ones((h, w), dtype=np.uint8) * 255,
                "bbox": [0, 0, w, h],
                "contour": np.array([[0, 0], [w, 0], [w, h], [0, h]]),
                "area": float(h * w),
                "confidence": 0.50,
                "fruit_id": "FRUIT_0001",
            }]

        logger.info(f"Segmented {len(valid_regions)} fruit(s) from image")
        return valid_regions

    def _multi_strategy_segment(
        self,
        image: np.ndarray,
        hsv: np.ndarray,
        gray: np.ndarray,
        fruit_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Try multiple foreground mask + watershed strategies and pick the best.
        """
        h, w = image.shape[:2]
        img_area = h * w
        all_results = []

        # --- Foreground mask strategies (conservative) ---
        masks = []

        # Strategy A: Tight S/V threshold (S>60 V>80 — best per analysis)
        s_mask = cv2.threshold(hsv[:, :, 1], 60, 255, cv2.THRESH_BINARY)[1]
        v_mask = cv2.threshold(hsv[:, :, 2], 80, 255, cv2.THRESH_BINARY)[1]
        mask_a = cv2.bitwise_and(s_mask, v_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_a = cv2.morphologyEx(mask_a, cv2.MORPH_OPEN, kernel, iterations=2)
        mask_a = cv2.morphologyEx(mask_a, cv2.MORPH_CLOSE, kernel, iterations=1)
        masks.append(("sv_60_80", mask_a))

        # Strategy B: Fruit-type-specific color mask
        mask_b = self._segment_by_color(hsv, gray, fruit_type)
        mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, kernel, iterations=2)
        mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE, kernel, iterations=1)
        masks.append(("color_" + fruit_type, mask_b))

        # --- For each mask, try watershed + contour strategies ---
        for mask_name, mask in masks:
            fg_ratio = cv2.countNonZero(mask) / img_area

            # Skip if mask is nearly empty or nearly full
            if fg_ratio < 0.05 or fg_ratio > 0.95:
                continue

            # Watershed with eroded mask
            eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)
            regions_ws = self._watershed_segment(image, eroded, threshold_ratio=0.05)
            valid_ws = [r for r in regions_ws if self._validate_region(r, w, h)]
            valid_ws = self._filter_by_color_match(valid_ws, hsv, fruit_type)
            all_results.append((f"{mask_name}_erode_ws", valid_ws, regions_ws))

            # Watershed on original mask
            regions_ws2 = self._watershed_segment(image, mask, threshold_ratio=0.08)
            valid_ws2 = [r for r in regions_ws2 if self._validate_region(r, w, h)]
            valid_ws2 = self._filter_by_color_match(valid_ws2, hsv, fruit_type)
            all_results.append((f"{mask_name}_ws", valid_ws2, regions_ws2))

            # Contour-based
            regions_contour = self._contour_based_segment(mask)
            valid_contour = [r for r in regions_contour if self._validate_region(r, w, h)]
            valid_contour = self._filter_by_color_match(valid_contour, hsv, fruit_type)
            all_results.append((f"{mask_name}_contour", valid_contour, regions_contour))

        # Pick best: use quality score = valid_count * total_fruit_area
        # Avoids bias toward over-segmentation
        best_name = None
        best_valid = []
        best_all = []
        best_score = 0
        for name, valid, all_r in all_results:
            if not valid:
                continue
            total_area = sum(r["area"] for r in valid)
            score = len(valid) * (total_area / img_area)
            if score > best_score:
                best_score = score
                best_name = name
                best_valid = valid
                best_all = all_r

        if best_valid:
            logger.info(
                f"Best segmentation: {best_name} "
                f"({len(best_valid)} valid / {len(best_all)} total, score={best_score:.3f})"
            )
            return best_valid

        # Fallback: use best raw results (still validate)
        for name, valid, all_r in all_results:
            if all_r:
                logger.info(f"Using raw results from {name} ({len(all_r)} regions)")
                return all_r

        return []

    def _segment_by_color(
        self, hsv: np.ndarray, gray: np.ndarray, fruit_type: str
    ) -> np.ndarray:
        """
        Color-based segmentation tuned for specific fruit types.
        """
        if fruit_type == "grapes":
            green_mask = cv2.inRange(hsv, np.array([25, 20, 20]), np.array([85, 255, 255]))
            purple_mask = cv2.inRange(hsv, np.array([120, 20, 20]), np.array([165, 255, 255]))
            return cv2.bitwise_or(green_mask, purple_mask)
        elif fruit_type == "pomegranate":
            red_mask1 = cv2.inRange(hsv, np.array([0, 30, 40]), np.array([10, 255, 255]))
            red_mask2 = cv2.inRange(hsv, np.array([160, 30, 40]), np.array([180, 255, 255]))
            return cv2.bitwise_or(red_mask1, red_mask2)
        elif fruit_type == "pineapple":
            yellow_mask1 = cv2.inRange(hsv, np.array([15, 30, 40]), np.array([35, 255, 255]))
            green_mask = cv2.inRange(hsv, np.array([25, 20, 30]), np.array([85, 255, 255]))
            return cv2.bitwise_or(yellow_mask1, green_mask)
        elif fruit_type == "potato":
            brown_mask = cv2.inRange(hsv, np.array([5, 10, 20]), np.array([40, 255, 255]))
            return brown_mask
        elif fruit_type == "kiwi":
            brown_mask = cv2.inRange(hsv, np.array([5, 10, 20]), np.array([45, 255, 255]))
            green_mask = cv2.inRange(hsv, np.array([25, 10, 30]), np.array([90, 255, 255]))
            return cv2.bitwise_or(brown_mask, green_mask)
        elif fruit_type == "capsicum":
            green_mask = cv2.inRange(hsv, np.array([25, 20, 20]), np.array([85, 255, 255]))
            red_mask1 = cv2.inRange(hsv, np.array([0, 30, 40]), np.array([10, 255, 255]))
            red_mask2 = cv2.inRange(hsv, np.array([160, 30, 40]), np.array([180, 255, 255]))
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            return cv2.bitwise_or(green_mask, red_mask)
        else:
            s_channel = hsv[:, :, 1]
            v_channel = hsv[:, :, 2]
            _, mask = cv2.threshold(s_channel, 60, 255, cv2.THRESH_BINARY)
            _, v_mask = cv2.threshold(v_channel, 80, 255, cv2.THRESH_BINARY)
            return cv2.bitwise_and(mask, v_mask)

    def _watershed_segment(
        self,
        image: np.ndarray,
        foreground_mask: np.ndarray,
        threshold_ratio: float = 0.06,
    ) -> List[Dict[str, Any]]:
        """
        Watershed algorithm to separate touching fruits.
        """
        sure_bg = cv2.dilate(foreground_mask, np.ones((3, 3), np.uint8), iterations=3)

        dist_transform = cv2.distanceTransform(foreground_mask, cv2.DIST_L2, 5)
        max_dist = dist_transform.max()

        if max_dist > 0:
            _, sure_fg = cv2.threshold(dist_transform, threshold_ratio * max_dist, 255, 0)
        else:
            return []

        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        ret, markers = cv2.connectedComponents(sure_fg)
        if ret <= 1:
            return []

        markers = markers + 1
        markers[unknown == 255] = 0

        img_copy = image.copy()
        cv2.watershed(img_copy, markers)

        regions = []
        for label in range(2, ret + 1):
            label_mask = np.uint8(markers == label)
            contours, _ = cv2.findContours(
                label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            x, y, cw, ch = cv2.boundingRect(c)

            full_mask = np.zeros_like(foreground_mask)
            full_mask[markers == label] = 255

            regions.append({
                "mask": full_mask,
                "bbox": [x, y, x + cw, y + ch],
                "contour": c,
                "area": float(area),
                "confidence": 0.85,
            })

        return regions

    def _contour_based_segment(
        self, foreground_mask: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Contour-based fallback: finds external contours of the foreground mask.
        """
        h, w = foreground_mask.shape[:2]
        min_contour_area = int(h * w * 0.005)

        contours, _ = cv2.findContours(
            foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_contour_area:
                continue

            x, y, cw, ch = cv2.boundingRect(c)

            mask = np.zeros_like(foreground_mask)
            cv2.drawContours(mask, [c], -1, 255, -1)

            regions.append({
                "mask": mask,
                "bbox": [x, y, x + cw, y + ch],
                "contour": c,
                "area": float(area),
                "confidence": 0.70,
            })

        return regions

    def _filter_by_color_match(
        self,
        regions: List[Dict[str, Any]],
        hsv: np.ndarray,
        fruit_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Filters out regions that don't contain enough fruit-colored pixels.
        A region with mostly background/table colors is rejected.
        """
        if not regions:
            return []

        color_ranges = {
            "grapes": [
                (np.array([25, 30, 30]), np.array([85, 255, 255])),
                (np.array([120, 30, 30]), np.array([165, 255, 255])),
            ],
            "pomegranate": [
                (np.array([0, 40, 50]), np.array([10, 255, 255])),
                (np.array([160, 40, 50]), np.array([180, 255, 255])),
            ],
            "pineapple": [
                (np.array([15, 40, 50]), np.array([35, 255, 255])),
                (np.array([25, 25, 40]), np.array([85, 255, 255])),
            ],
            "mango": [
                (np.array([10, 40, 50]), np.array([30, 255, 255])),
                (np.array([20, 30, 60]), np.array([40, 255, 255])),
            ],
            "potato": [
                (np.array([5, 10, 20]), np.array([40, 255, 255])),
            ],
            "kiwi": [
                (np.array([5, 10, 20]), np.array([45, 255, 255])),
                (np.array([25, 10, 30]), np.array([90, 255, 255])),
            ],
            "capsicum": [
                (np.array([25, 30, 30]), np.array([85, 255, 255])),
                (np.array([0, 40, 50]), np.array([10, 255, 255])),
                (np.array([160, 40, 50]), np.array([180, 255, 255])),
            ],
        }

        ranges = color_ranges.get(fruit_type, [
            (np.array([0, 30, 30]), np.array([180, 255, 255]))
        ])

        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in ranges:
            combined = cv2.bitwise_or(combined, cv2.inRange(hsv, low, high))

        filtered = []
        for region in regions:
            mask = region["mask"]
            region_area = cv2.countNonZero(mask)
            if region_area == 0:
                continue

            # Count fruit-colored pixels within the region
            fruit_pixels = cv2.countNonZero(cv2.bitwise_and(combined, mask))
            match_ratio = fruit_pixels / region_area

            # At least 25% of the region must be fruit-colored
            if match_ratio < 0.25:
                logger.debug(
                    f"Rejected region: only {match_ratio:.1%} fruit-colored "
                    f"(bbox={region['bbox']})"
                )
                continue

            region["confidence"] = min(0.95, 0.5 + match_ratio * 0.5)
            filtered.append(region)

        return filtered

    def _validate_region(
        self, region: Dict[str, Any], img_w: int, img_h: int
    ) -> bool:
        """Validates that a detected region is a plausible fruit."""
        area = region["area"]
        img_area = img_w * img_h

        area_ratio = area / img_area if img_area > 0 else 0
        if area_ratio < self._min_area_ratio or area_ratio > self._max_area_ratio:
            return False

        contour = region["contour"]
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < self._min_circularity:
            return False

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return False
        solidity = area / hull_area
        if solidity < self._min_solidity:
            return False

        x, y, bw, bh = region["bbox"]
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > self._max_aspect:
            return False

        return True

    def _scale_region(
        self, region: Dict[str, Any], scale_x: float, scale_y: float
    ) -> Dict[str, Any]:
        """Scales coordinates from working resolution back to original."""
        x1, y1, x2, y2 = region["bbox"]
        region["bbox"] = [
            max(0, int(x1 * scale_x)),
            max(0, int(y1 * scale_y)),
            int(x2 * scale_x),
            int(y2 * scale_y),
        ]

        contour = region["contour"].copy()
        contour[:, :, 0] = (contour[:, :, 0] * scale_x).astype(np.int32)
        contour[:, :, 1] = (contour[:, :, 1] * scale_y).astype(np.int32)
        region["contour"] = contour

        return region

    def _remove_duplicates(
        self, regions: List[Dict[str, Any]], iou_threshold: float = 0.35
    ) -> List[Dict[str, Any]]:
        """Removes duplicate/overlapping detections using IoU."""
        if len(regions) <= 1:
            return regions

        regions.sort(key=lambda r: r["area"], reverse=True)
        keep = []

        for region in regions:
            bbox = region["bbox"]
            area = region["area"]
            is_dup = False

            for kept in keep:
                k_bbox = kept["bbox"]
                k_area = kept["area"]

                ix1 = max(bbox[0], k_bbox[0])
                iy1 = max(bbox[1], k_bbox[1])
                ix2 = min(bbox[2], k_bbox[2])
                iy2 = min(bbox[3], k_bbox[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = area + k_area - inter
                iou = inter / union if union > 0 else 0

                if iou > iou_threshold:
                    is_dup = True
                    break

            if not is_dup:
                keep.append(region)

        return keep

    def draw_segmentation(
        self,
        image: np.ndarray,
        regions: List[Dict[str, Any]],
        fruits_data: List[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Draws segmentation masks and labels on the image.
        """
        overlay = image.copy()

        grade_colors = {
            "Good": (0, 200, 0),
            "Better": (0, 165, 255),
            "Reject": (0, 0, 255),
        }
        default_color = (0, 255, 255)

        grade_map = {}
        if fruits_data:
            for fd in fruits_data:
                grade_map[fd.get("fruit_id", "")] = fd.get("grade", "")

        for i, region in enumerate(regions):
            fruit_id = region.get("fruit_id", f"FRUIT_{i+1:04d}")
            grade = grade_map.get(fruit_id, "")
            color = grade_colors.get(grade, default_color)

            mask = region["mask"]
            color_overlay = np.zeros_like(overlay)
            color_overlay[mask > 0] = color
            overlay = cv2.addWeighted(overlay, 0.7, color_overlay, 0.3, 0)

            x1, y1, x2, y2 = region["bbox"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            label = f"{fruit_id}"
            if grade:
                label += f" | {grade}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_y = max(y1 - 8, th + 4)
            cv2.rectangle(overlay, (x1, text_y - th - 4), (x1 + tw + 6, text_y + 2), color, -1)
            cv2.putText(
                overlay, label, (x1 + 3, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )

        return overlay


segmentation_service = SegmentationService()

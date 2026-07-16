"""
Tiling Module — Overlapping Tile Inference for Small-Object Detection

WHY TILING IMPROVES SMALL-OBJECT DETECTION
============================================
YOLO models internally resize the input image to a fixed resolution (e.g. 640×640
or 1280×1280). In a 4K drone video (3840×2160), a typical car occupies roughly
40×20 pixels — about 0.02% of the frame area. After YOLO's internal downsampling
to 640×640, that car shrinks to ≈7×4 pixels, which is below the effective
receptive field of the network's anchor boxes, causing many misses.

Tiling solution:
  • Split the full frame into N overlapping tiles (e.g. 2×2 or 3×3 grid).
  • Each tile covers 1/N of the frame, so after YOLO downsampling, the same car
    now occupies ≈14×8 pixels (2×2 grid) — well within detection range.
  • Overlap prevents objects at tile boundaries from being cut in half.
  • A final class-aware NMS step merges all tile detections into one clean list.

Design decisions:
  • Overlap fraction of 0.2 (20%) balances redundant computation vs. missed edges.
  • Class-aware NMS avoids suppressing a car adjacent to a motorcycle.
  • Results from the full-frame pass are also included so large objects (buses,
    trucks) that span tile boundaries are not lost.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tile generation
# ---------------------------------------------------------------------------

def create_tiles(
    frame: np.ndarray,
    grid: Tuple[int, int] = (2, 2),
    overlap: float = 0.2,
) -> List[Dict]:
    """
    Partition a frame into an overlapping grid of sub-images (tiles).

    Args:
        frame:   BGR image as H×W×C numpy array.
        grid:    (rows, cols) — number of tiles along each axis.
        overlap: Fractional padding added to each tile edge (0.0–0.75).

    Returns:
        List of tile dicts, each containing:
            tile      — cropped numpy array (BGR)
            x_offset  — left-edge pixel in the original frame
            y_offset  — top-edge pixel in the original frame
    """
    H, W = frame.shape[:2]
    rows, cols = grid

    step_y = H / rows
    step_x = W / cols
    pad_y = int(step_y * overlap)
    pad_x = int(step_x * overlap)

    tiles: List[Dict] = []
    for r in range(rows):
        for c in range(cols):
            y1 = max(0, int(r * step_y) - pad_y)
            y2 = min(H, int((r + 1) * step_y) + pad_y)
            x1 = max(0, int(c * step_x) - pad_x)
            x2 = min(W, int((c + 1) * step_x) + pad_x)

            tiles.append({
                "tile":     frame[y1:y2, x1:x2],
                "x_offset": x1,
                "y_offset": y1,
            })

    logger.debug("Created %d tiles from %dx%d frame (grid=%s, overlap=%.2f)",
                 len(tiles), W, H, grid, overlap)
    return tiles


def remap_detections_to_frame(
    detections: List[Dict],
    x_offset: int,
    y_offset: int,
) -> List[Dict]:
    """
    Translate bounding-box coordinates from tile space back to full-frame space.

    Args:
        detections: Detection dicts with 'bbox' = [x1, y1, x2, y2] in tile coords.
        x_offset:   Pixel offset of tile's left edge in the full frame.
        y_offset:   Pixel offset of tile's top edge in the full frame.

    Returns:
        New list of detection dicts with updated bbox coordinates.
    """
    remapped: List[Dict] = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        remapped.append({
            **det,
            "bbox": [
                x1 + x_offset,
                y1 + y_offset,
                x2 + x_offset,
                y2 + y_offset,
            ],
        })
    return remapped


# ---------------------------------------------------------------------------
# Class-aware NMS
# ---------------------------------------------------------------------------

def nms(detections: List[Dict], iou_threshold: float = 0.45) -> List[Dict]:
    """
    Class-aware Non-Maximum Suppression to remove duplicate detections
    produced by overlapping tiles.

    WHY CLASS-AWARE?
    Suppression only happens between detections of the *same* class.
    A high-IoU (car, motorcycle) pair will NOT suppress each other, which
    prevents losing a two-wheeler parked right next to a car.

    Args:
        detections:    Merged list of all detections (across tiles + full frame).
        iou_threshold: Boxes with IoU > threshold and same class are duplicates.

    Returns:
        Deduplicated detection list, best-confidence box kept per group.
    """
    if not detections:
        return []

    # Sort by confidence descending so the best box is always picked first.
    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)

    keep: List[Dict] = []
    while detections:
        best = detections.pop(0)
        keep.append(best)

        remaining: List[Dict] = []
        for det in detections:
            # Only suppress within the same class.
            if det["class_id"] != best["class_id"]:
                remaining.append(det)
                continue
            if _iou(best["bbox"], det["bbox"]) < iou_threshold:
                remaining.append(det)
        detections = remaining

    return keep


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / (union + 1e-7)

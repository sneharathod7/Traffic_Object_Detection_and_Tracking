"""
Detection Module — SAHI Sliced Inference with RT-DETR Backend

This module implements sliced inference using the SAHI framework with an RT-DETR
object detection model backend. This setup is optimized for detecting small
objects in dense aerial drone traffic imagery.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — COCO class IDs for traffic objects
# ---------------------------------------------------------------------------

TRAFFIC_CLASSES: Dict[int, str] = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def _auto_device() -> str:
    """Return the best available inference device."""
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# SahiRTDetrDetector class
# ---------------------------------------------------------------------------

class SahiRTDetrDetector:
    """
    RT-DETR detector wrapper with SAHI sliced inference.
    
    Typical usage::

        detector = SahiRTDetrDetector(model_path="rtdetr-l.pt")
        detections = detector.detect(frame)
        # detections = [{"bbox": [x1, y1, x2, y2], "confidence": 0.73,
        #                "class_id": 2, "class_name": "car"}, ...]
    """

    def __init__(
        self,
        model_path: str = "rtdetr-l.pt",
        slice_height: int = 256,
        slice_width: int = 256,
        overlap_height_ratio: float = 0.75,
        overlap_width_ratio: float = 0.75,
        conf: float = 0.10,
        device: Optional[str] = None,
    ) -> None:
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.conf = conf
        self.device = device or _auto_device()
        self.target_classes = list(TRAFFIC_CLASSES.keys())

        logger.info(
            "Loading SAHI RT-DETR model '%s' on device '%s'",
            model_path,
            self.device,
        )

        # Load RT-DETR model via SAHI's AutoDetectionModel wrapper
        self.model = AutoDetectionModel.from_pretrained(
            model_type="rtdetr",
            model_path=model_path,
            confidence_threshold=self.conf,
            device=self.device,
        )
        logger.info(
            "SAHI RT-DETR Detector loaded. slice_size=(%dx%d), overlap=(%.2f, %.2f), conf=%.2f",
            slice_height,
            slice_width,
            overlap_height_ratio,
            overlap_width_ratio,
            conf,
        )

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect all traffic objects in *frame* using SAHI sliced inference.

        Args:
            frame: BGR numpy array (H × W × 3).

        Returns:
            Filtered list of detection dicts:
            [{"bbox": [x1, y1, x2, y2], "confidence": float,
              "class_id": int, "class_name": str}, ...]
        """
        # Run SAHI sliced prediction
        results = get_sliced_prediction(
            frame,
            self.model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            verbose=0,  # Turn off print messages from SAHI
        )

        detections: List[Dict] = []
        for pred in results.object_prediction_list:
            class_id = int(pred.category.id)
            if class_id not in TRAFFIC_CLASSES:
                continue

            x1, y1, x2, y2 = pred.bbox.to_xyxy()
            confidence = float(pred.score.value)

            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": confidence,
                "class_id": class_id,
                "class_name": TRAFFIC_CLASSES[class_id],
            })

        logger.debug(
            "SAHI detections count: raw=%d, matched_traffic=%d",
            len(results.object_prediction_list),
            len(detections),
        )
        return detections

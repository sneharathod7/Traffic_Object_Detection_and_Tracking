"""
Detection Module — YOLOv8 with High-Resolution Tiling Inference

DESIGN DECISIONS
================
Model choice — YOLOv8m / YOLOv8l:
    The nano (n) and small (s) variants trade accuracy for speed.
    In a dense Indian intersection every missed detection is a missed road-user.
    The medium (m) variant gives ≈5–8 mAP points more than nano with only 2×
    the inference time, which is an excellent accuracy-speed tradeoff for batch
    video processing where real-time is not strictly required.

Input resolution — imgsz=1280:
    COCO pre-training used 640. Doubling to 1280 keeps small objects (pedestrians,
    motorcycles) at a larger fraction of the network's input, recovering the small
    objects that the default resolution would miss due to downsampling.

Test-Time Augmentation (augment=True):
    TTA runs inference on the image and several flipped/scaled variants and
    merges the results. It typically adds 2–3 mAP at the cost of 3–5× inference
    time. Useful for offline batch processing; disable for real-time.

Confidence threshold — conf=0.25:
    Low enough to catch partially occluded or distant objects that ByteTrack's
    second-stage low-confidence matching will later confirm with track history.

IoU threshold for NMS — iou=0.5:
    Balanced between suppressing duplicates (lower value) and keeping adjacent
    objects in dense clusters (higher value).

GPU / MPS / CPU auto-selection:
    Checked at init time so the same script runs on a Mac (MPS), Linux (CUDA),
    or CPU-only CI machine without code changes.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO, RTDETR

from tiling import create_tiles, nms, remap_detections_to_frame

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
# Detector class
# ---------------------------------------------------------------------------

class Detector:
    """
    YOLOv8 / RT-DETR detector with optional tiling for small-object detection.

    Typical usage::

        detector = Detector(model_path="yolov8m.pt", use_tiling=True)
        detections = detector.detect(frame)
        # detections = [{"bbox": [x1,y1,x2,y2], "confidence": 0.73,
        #                "class_id": 2, "class_name": "car"}, ...]
    """

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        imgsz: int = 1280,
        conf: float = 0.25,
        iou: float = 0.5,
        use_tiling: bool = True,
        tile_grid: Tuple[int, int] = (2, 2),
        tile_overlap: float = 0.2,
        use_tta: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.use_tiling = use_tiling
        self.tile_grid = tile_grid
        self.tile_overlap = tile_overlap
        self.use_tta = use_tta
        self.device = device or _auto_device()
        self.target_classes = list(TRAFFIC_CLASSES.keys())

        # Calibrated class-specific confidence thresholds
        self.class_thresholds = {
            "person": 0.25,
            "motorcycle": 0.30,
            "car": 0.35,
            "bus": 0.40,
            "truck": 0.45,
        }
        if conf != 0.25:
            # Shift relative to the requested global confidence
            diff = conf - 0.25
            self.class_thresholds = {
                k: max(0.01, v + diff) for k, v in self.class_thresholds.items()
            }

        if "rtdetr" in model_path.lower():
            logger.info("Loading RT-DETR model '%s' on device '%s'", model_path, self.device)
            self.model = RTDETR(model_path)
        else:
            logger.info("Loading YOLOv8 model '%s' on device '%s'", model_path, self.device)
            self.model = YOLO(model_path)

        self.model.to(self.device)
        logger.info("Model loaded. imgsz=%d, conf=%.2f, iou=%.2f, tiling=%s",
                    imgsz, conf, iou, use_tiling)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_yolo(self, image: np.ndarray) -> List[Dict]:
        """
        Run detection on a single image patch (full frame or tile).

        Returns raw detections in the canonical dict format.
        """
        with torch.no_grad():
            inference_args = {
                "imgsz": self.imgsz,
                "conf": min(self.class_thresholds.values()),
                "iou": self.iou,
                "classes": self.target_classes,
                "verbose": False,
                "device": self.device,
            }
            if isinstance(self.model, YOLO):
                inference_args["augment"] = self.use_tta
            
            results = self.model(image, **inference_args)

        detections: List[Dict] = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id not in TRAFFIC_CLASSES:
                    continue
                confidence = float(box.conf[0])
                class_name = TRAFFIC_CLASSES[class_id]

                # Filter by class-calibrated threshold
                if confidence < self.class_thresholds.get(class_name, self.conf):
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                detections.append({
                    "bbox":       [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": confidence,
                    "class_id":   class_id,
                    "class_name": class_name,
                })
        return detections

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect all traffic objects in *frame*, optionally using tiling.

        Pipeline when tiling is enabled:
          1. Run YOLOv8 on the full frame  → catches large objects.
          2. Split frame into overlapping tiles.
          3. Run YOLOv8 on each tile       → catches small objects.
          4. Remap tile-space boxes to frame coordinates.
          5. Merge all detections and apply class-aware NMS.

        Args:
            frame: BGR numpy array (H × W × 3).

        Returns:
            Filtered, deduplicated list of detection dicts:
            [{"bbox": [x1,y1,x2,y2], "confidence": float,
              "class_id": int, "class_name": str}, ...]
        """
        # Always run the full-frame pass to capture large vehicles.
        full_dets = self._run_yolo(frame)

        if not self.use_tiling:
            logger.debug("Tiling disabled. Full-frame detections: %d", len(full_dets))
            return full_dets

        # Tile inference for small objects.
        tiles = create_tiles(frame, grid=self.tile_grid, overlap=self.tile_overlap)
        tile_dets: List[Dict] = []
        for tile_info in tiles:
            raw = self._run_yolo(tile_info["tile"])
            remapped = remap_detections_to_frame(
                raw, tile_info["x_offset"], tile_info["y_offset"]
            )
            tile_dets.extend(remapped)

        all_dets = full_dets + tile_dets
        final = nms(all_dets, iou_threshold=self.iou, cross_class_iou_threshold=0.70)

        logger.debug(
            "Detections — full=%d  tiles=%d  after_NMS=%d",
            len(full_dets), len(tile_dets), len(final),
        )
        return final

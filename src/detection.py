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

DEFAULT_CLASS_THRESHOLDS: Dict[str, float] = {
    "truck": 0.65,
    "bus": 0.60,
    "car": 0.35,
    "motorcycle": 0.30,
    "person": 0.35,
}

CLASS_PRIORITY: Dict[str, int] = {
    "motorcycle": 0,
    "car": 1,
    "person": 2,
    "bus": 3,
    "truck": 4,
}


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


def resolve_cross_class_overlaps(detections: List[Dict], iou_threshold: float = 0.45) -> List[Dict]:
    """
    Resolve overlapping detections of different classes on the same object.
    
    Greedily keeps detections belonging to higher-priority classes, suppressing
    overlapping lower-priority classes (e.g. keeping 'car' over 'truck' / 'bus').
    """
    if not detections:
        return []

    # Sort detections so that higher priority classes (lower priority index) come first.
    # For same priority class, sort by confidence descending.
    sorted_dets = sorted(
        detections,
        key=lambda d: (CLASS_PRIORITY.get(d["class_name"], 99), -d["confidence"])
    )

    keep: List[Dict] = []
    while sorted_dets:
        best = sorted_dets.pop(0)
        keep.append(best)

        remaining: List[Dict] = []
        for det in sorted_dets:
            if _iou(best["bbox"], det["bbox"]) >= iou_threshold:
                # Overlap exceeds threshold: suppress the lower priority box
                continue
            remaining.append(det)
        sorted_dets = remaining

    return keep


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
        class_thresholds: Optional[Dict[str, float]] = None,
        tracker_high_thresh: float = 0.50,
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
        self.tracker_high_thresh = tracker_high_thresh

        # Initialize class-specific thresholds
        self.class_thresholds = dict(DEFAULT_CLASS_THRESHOLDS)
        if class_thresholds:
            self.class_thresholds.update(class_thresholds)

        # Run inference with the minimum needed threshold to catch all candidate classes
        self.model_conf = min(self.conf, min(self.class_thresholds.values()))

        if "rtdetr" in model_path.lower():
            logger.info("Loading RT-DETR model '%s' on device '%s'", model_path, self.device)
            self.model = RTDETR(model_path)
        else:
            logger.info("Loading YOLOv8 model '%s' on device '%s'", model_path, self.device)
            self.model = YOLO(model_path)

        self.model.to(self.device)
        logger.info(
            "Model loaded. imgsz=%d, conf=%.2f, model_conf=%.2f, iou=%.2f, tiling=%s, thresholds=%s",
            imgsz, conf, self.model_conf, iou, use_tiling, self.class_thresholds
        )

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
                "conf": self.model_conf,
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

                # 1. Class-aware confidence filtering
                thresh = self.class_thresholds.get(class_name, self.conf)
                if confidence < thresh:
                    continue

                # 2. Confidence mapping/boosting for tracker alignment
                mapped_conf = confidence
                if thresh < self.tracker_high_thresh:
                    # Map [thresh, 1.0] -> [tracker_high_thresh, 1.0]
                    mapped_conf = self.tracker_high_thresh + (confidence - thresh) * (1.0 - self.tracker_high_thresh) / (1.0 - thresh + 1e-7)

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                detections.append({
                    "bbox":       [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": mapped_conf,
                    "raw_confidence": confidence,
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
          1. Run YOLOv8 / RT-DETR on the full frame  → catches large objects.
          2. Split frame into overlapping tiles.
          3. Run detector on each tile       → catches small objects.
          4. Remap tile-space boxes to frame coordinates.
          5. Merge all detections and apply class-aware NMS.
          6. Apply cross-class overlap resolution to suppress redundant classes.

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
            final = full_dets
        else:
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
            final = nms(all_dets, iou_threshold=self.iou)

        # Apply cross-class overlap resolution to suppress redundant/fighting classes
        resolved = resolve_cross_class_overlaps(final, iou_threshold=0.45)

        logger.debug(
            "Detections — full=%d  tiles=%d  after_NMS=%d  after_resolution=%d",
            len(full_dets), len(tile_dets) if self.use_tiling else 0, len(final), len(resolved),
        )
        return resolved

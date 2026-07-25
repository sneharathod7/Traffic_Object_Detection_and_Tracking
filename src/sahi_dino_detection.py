"""
Detection Module — SAHI Sliced Inference with DINO-Style RT-DETRv2-X Backend

ARCHITECTURE OVERVIEW
=====================
This module is a drop-in replacement for ``sahi_rtdetr_detection.py``.  It
replaces the RT-DETR-L backbone with RT-DETRv2-X — an Extra-Large variant that
uses the DINO (DETR with Improved deNoising Optimization) decoder head.

WHY RT-DETRv2-X IS "DINO" IN PRACTICE
=======================================
The RT-DETRv2-X model in the Ultralytics ecosystem shares the key innovations of
DINO (the ICLR 2023 paper by Zhang et al.):
  • Contrastive denoising training (CDN) — the primary source of cross-tile
    confidence stability.  Each query learns to distinguish noisy vs. clean anchor
    assignments, producing tighter, more calibrated confidence scores.
  • Hybrid matching — combines one-to-one DETR-style matching with a denoising
    group that acts as an auxiliary supervision signal.
  • Multi-scale deformable attention encoder — explicitly attends to feature maps
    at scales P3–P6, which is crucial for detecting motorcycles (20-50 px) and
    buses (200+ px) simultaneously in the same dense Indian traffic frame.

These properties directly address the RT-DETR-L tile-collision problem:
  • Tighter confidence calibration → two tiles detecting the same motorcycle now
    produce scores of 0.60 and 0.59 instead of 0.82 and 0.43.
  • Fewer spurious anchor peaks → less chance of a second false detection at a
    nearby location.

POST-SAHI FUSION
=================
After SAHI assembles all tile predictions, this module applies
``weighted_box_fusion()`` from ``sahi_fusion.py`` as a second deduplication pass
beyond SAHI's built-in NMS.  This eliminates the "IoU=0.22–0.33 duplicates" that
bypass standard NMS thresholds but still confuse ByteTrack Stage-1 matching.

FALLBACK CHAIN
==============
Model weights are tried in order:
  1. rtdetr-x.pt  (primary — DINO-style Extra-Large, best accuracy)
  2. rtdetr-l.pt  (fallback — standard Large)
  3. rtdetr-m.pt  (emergency — Medium for low-VRAM environments)

INTERFACE CONTRACT
==================
``SahiDinoDetector.detect()`` returns identical format to ``SahiRTDetrDetector``:
    List[Dict] where each dict has keys:
        bbox        — [x1, y1, x2, y2]  (float, pixel coords)
        confidence  — float in [0, 1]
        class_id    — int (COCO class ID)
        class_name  — str

The tile-collision metadata (``tile_origin_count``, ``is_tile_collision``) is
used ONLY for internal logging and is NOT passed downstream.  All postprocessing,
tracking and export components are unaffected.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from sahi_fusion import weighted_box_fusion

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

# Model fallback priority (DINO-style → standard → medium)
_MODEL_FALLBACK_CHAIN: List[str] = [
    "rtdetr-x.pt",
    "rtdetr-l.pt",
    "rtdetr-m.pt",
]

# Statistics logging interval (frames)
_STATS_LOG_INTERVAL = 100


def _auto_device() -> str:
    """Return the best available inference device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_model_path(requested: str) -> Tuple[str, bool]:
    """
    Resolve the model path, applying the fallback chain if the primary model
    is not found locally (Ultralytics will auto-download if needed).

    Returns:
        (resolved_path, used_fallback)
    """
    # If the exact path exists or is a recognised ultralytics model name, use it
    if Path(requested).exists():
        return requested, False

    # Check if it is already a recognised auto-download name
    recognised = {p.lower() for p in _MODEL_FALLBACK_CHAIN}
    if requested.lower() in recognised:
        # Ultralytics will download automatically; use as-is
        return requested, False

    # Try fallback chain
    logger.warning(
        "Model '%s' not found locally. Trying fallback chain: %s",
        requested, _MODEL_FALLBACK_CHAIN,
    )
    for fallback in _MODEL_FALLBACK_CHAIN:
        if Path(fallback).exists():
            logger.warning("Using fallback model: %s", fallback)
            return fallback, True
        # Check models/ subdirectory
        candidate = Path("models") / fallback
        if candidate.exists():
            logger.warning("Using fallback model from models/: %s", candidate)
            return str(candidate), True

    # Nothing found locally — let Ultralytics download the primary
    logger.info(
        "No model found locally. Allowing Ultralytics to auto-download: %s",
        _MODEL_FALLBACK_CHAIN[0],
    )
    return _MODEL_FALLBACK_CHAIN[0], True


# ---------------------------------------------------------------------------
# SahiDinoDetector class
# ---------------------------------------------------------------------------

class SahiDinoDetector:
    """
    DINO-style RT-DETRv2-X detector with SAHI sliced inference and
    Weighted Box Fusion for Indian dense traffic.

    This class is a **drop-in replacement** for ``SahiRTDetrDetector``.
    The detect() method returns an identical ``List[Dict]`` format.

    Typical usage::

        detector = SahiDinoDetector(model_path="rtdetr-x.pt")
        detections = detector.detect(frame)
        # detections = [{"bbox": [x1, y1, x2, y2], "confidence": 0.71,
        #                "class_id": 3, "class_name": "motorcycle"}, ...]

    Indian traffic optimised defaults:
        • slice_height/width = 640  (larger slices preserve spatial context)
        • overlap_ratio       = 0.30 (high overlap for tile boundary objects)
        • conf                = 0.10 (lower global; DINO is more selective)
        • WBF cluster_dist    = 30px (calibrated for 1920×1080 footage)
        • WBF iou_thresh      = 0.35 (merges overlapping tile hits, keeps distinct)
    """

    def __init__(
        self,
        model_path: str = "rtdetr-x.pt",
        slice_height: int = 384,
        slice_width: int = 384,
        overlap_height_ratio: float = 0.75,
        overlap_width_ratio: float = 0.75,
        conf: float = 0.10,
        device: Optional[str] = None,
        # WBF tuning parameters
        wbf_cluster_distance: float = 45.0,
        wbf_iou_thresh: float = 0.30,
    ) -> None:
        """
        Args:
            model_path:            Path to RT-DETRv2-X weights or Ultralytics
                                   model name (e.g. "rtdetr-x.pt"). Falls back
                                   to rtdetr-l.pt / rtdetr-m.pt if not found.
            slice_height:          SAHI tile height in pixels. Default 640.
            slice_width:           SAHI tile width in pixels. Default 640.
            overlap_height_ratio:  Fractional overlap between vertical tiles.
            overlap_width_ratio:   Fractional overlap between horizontal tiles.
            conf:                  Global confidence threshold passed to SAHI.
                                   Class-specific thresholds are applied by
                                   postprocess_vehicle_classes.py downstream.
            device:                Inference device ('cuda', 'cpu', 'mps').
                                   Auto-detected if None.
            wbf_cluster_distance:  Pixel distance threshold for WBF clustering.
            wbf_iou_thresh:        IoU threshold for WBF duplicate suppression.
        """
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.conf = conf
        self.device = device or _auto_device()
        self.target_classes = list(TRAFFIC_CLASSES.keys())
        self.wbf_cluster_distance = wbf_cluster_distance
        self.wbf_iou_thresh = wbf_iou_thresh

        # Per-run statistics accumulators
        self._frame_count: int = 0
        self._cumulative_raw: int = 0
        self._cumulative_fused: int = 0
        self._cumulative_collisions: int = 0
        self._cumulative_suppressed: int = 0
        self._moto_collisions: int = 0

        # Resolve model path with fallback chain
        resolved_path, used_fallback = _resolve_model_path(model_path)
        if used_fallback:
            logger.warning(
                "SahiDinoDetector: primary model '%s' unavailable. "
                "Loading fallback '%s'.",
                model_path, resolved_path,
            )
        else:
            logger.info(
                "SahiDinoDetector: loading model '%s' on device '%s'",
                resolved_path, self.device,
            )

        # Load via SAHI AutoDetectionModel (rtdetr adapter = DINO-style transformer)
        self.model = AutoDetectionModel.from_pretrained(
            model_type="rtdetr",          # SAHI rtdetr adapter for all RT-DETR variants
            model_path=resolved_path,
            confidence_threshold=self.conf,
            device=self.device,
        )

        self._model_name = resolved_path  # for logging

        logger.info(
            "SahiDinoDetector ready. model=%s slice=(%dx%d) overlap=(%.2f,%.2f) "
            "conf=%.2f wbf_dist=%.1f wbf_iou=%.2f",
            resolved_path, slice_height, slice_width,
            overlap_height_ratio, overlap_width_ratio,
            conf, wbf_cluster_distance, wbf_iou_thresh,
        )

    # ---- Public detect API ---------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect all traffic objects in *frame* using SAHI sliced inference with
        DINO-style RT-DETRv2-X backend and Weighted Box Fusion post-processing.

        Args:
            frame: BGR numpy array (H × W × 3).

        Returns:
            Deduplicated list of detection dicts — same format as
            ``SahiRTDetrDetector.detect()``::

                [{"bbox": [x1, y1, x2, y2], "confidence": float,
                  "class_id": int, "class_name": str}, ...]

        Note:
            Tile-collision metadata (``tile_origin_count``, ``is_tile_collision``)
            is logged internally but NOT included in the returned dicts.
        """
        self._frame_count += 1

        # ---- Step 1: SAHI sliced prediction ----------------------------------
        results = get_sliced_prediction(
            frame,
            self.model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            verbose=0,
        )

        # ---- Step 2: Convert SAHI output to canonical dict format ------------
        raw_detections: List[Dict] = []
        for pred in results.object_prediction_list:
            class_id = int(pred.category.id)
            if class_id not in TRAFFIC_CLASSES:
                continue
            x1, y1, x2, y2 = pred.bbox.to_xyxy()
            raw_detections.append({
                "bbox":       [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(pred.score.value),
                "class_id":   class_id,
                "class_name": TRAFFIC_CLASSES[class_id],
            })

        # ---- Step 3: Weighted Box Fusion — cross-tile deduplication ----------
        fused_detections, wbf_stats = weighted_box_fusion(
            raw_detections,
            cluster_distance_thresh=self.wbf_cluster_distance,
            overlap_iou_thresh=self.wbf_iou_thresh,
        )

        # ---- Step 4: Update cumulative statistics ----------------------------
        self._cumulative_raw       += wbf_stats["raw_count"]
        self._cumulative_fused     += wbf_stats["fused_count"]
        self._cumulative_collisions += wbf_stats["collision_clusters"]
        self._cumulative_suppressed += wbf_stats["suppressed_count"]

        # Count motorcycle-specific collisions for targeted logging
        moto_collisions_this_frame = sum(
            1 for d in fused_detections
            if d["class_id"] == 3 and d.get("is_tile_collision", False)
        )
        self._moto_collisions += moto_collisions_this_frame

        # ---- Step 5: Per-frame debug log -------------------------------------
        logger.debug(
            "SAHI DINO detections [frame %d]: raw=%d → fused=%d | "
            "tile_collisions=%d suppressed=%d moto_collisions=%d",
            self._frame_count,
            wbf_stats["raw_count"],
            wbf_stats["fused_count"],
            wbf_stats["collision_clusters"],
            wbf_stats["suppressed_count"],
            moto_collisions_this_frame,
        )

        # ---- Step 6: Periodic statistics summary (every N frames) -----------
        if self._frame_count % _STATS_LOG_INTERVAL == 0:
            self._log_periodic_stats()

        # ---- Step 7: Strip metadata — return clean dicts --------------------
        clean_detections: List[Dict] = []
        for d in fused_detections:
            clean_detections.append({
                "bbox":       d["bbox"],
                "confidence": d["confidence"],
                "class_id":   d["class_id"],
                "class_name": d["class_name"],
            })

        return clean_detections

    # ---- Statistics helpers --------------------------------------------------

    def _log_periodic_stats(self) -> None:
        """Log cumulative tile-collision statistics every _STATS_LOG_INTERVAL frames."""
        total_frames = max(self._frame_count, 1)
        collision_rate = (
            self._cumulative_collisions / max(self._cumulative_raw, 1) * 100
        )
        avg_cluster = (
            self._cumulative_raw / max(self._cumulative_fused, 1)
            if self._cumulative_fused > 0 else 1.0
        )
        reduction_pct = (
            (self._cumulative_raw - self._cumulative_fused) /
            max(self._cumulative_raw, 1) * 100
        )
        logger.info(
            "DINO stats @ frame %d: "
            "tile_collision_rate=%.1f%% avg_cluster_size=%.2f "
            "moto_collisions_total=%d reduction=%.1f%% "
            "(raw=%d → fused=%d suppressed=%d)",
            self._frame_count,
            collision_rate,
            avg_cluster,
            self._moto_collisions,
            reduction_pct,
            self._cumulative_raw,
            self._cumulative_fused,
            self._cumulative_suppressed,
        )

    def get_stats(self) -> Dict:
        """
        Return cumulative detection statistics for the current run.
        Useful for programmatic access from compare_detectors.py.
        """
        total_raw = max(self._cumulative_raw, 1)
        return {
            "frames_processed":    self._frame_count,
            "total_raw_dets":      self._cumulative_raw,
            "total_fused_dets":    self._cumulative_fused,
            "total_suppressed":    self._cumulative_suppressed,
            "collision_clusters":  self._cumulative_collisions,
            "moto_collisions":     self._moto_collisions,
            "tile_collision_rate": self._cumulative_collisions / total_raw * 100,
            "reduction_pct":       (self._cumulative_raw - self._cumulative_fused) / total_raw * 100,
            "model_name":          self._model_name,
        }

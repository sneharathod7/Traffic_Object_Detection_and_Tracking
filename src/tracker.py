"""
ByteTrack — Multi-Object Tracker for Dense Traffic Scenes

HOW BYTETRACK WORKS
===================
Traditional trackers (SORT, DeepSORT) only use *high-confidence* detections
for track association. In crowded scenes this discards many real objects that
happen to have lower detection scores due to partial occlusion — exactly what
happens at a busy Indian intersection.

ByteTrack's key innovation: use EVERY detection box, including low-confidence
ones, through a two-stage matching cascade:

  Stage 1 — High-confidence ↔ All active tracks
    Associate high-conf detections (score ≥ high_thresh) with all currently
    Tracked and recently-Lost tracks using IoU cost + Hungarian assignment.
    This handles the majority of clear, unoccluded objects.

  Stage 2 — Low-confidence ↔ Unmatched tracked tracks
    Associate low-conf detections (low_thresh ≤ score < high_thresh) with the
    tracks that were NOT matched in Stage 1. Low-confidence boxes often represent
    partially occluded objects; their temporal context (existing track history)
    provides the evidence needed to confirm the association.

  New tracks
    Unmatched high-confidence detections that survive both stages start new
    tentative tracks (confirmed after min_hits consecutive matches).

  Track removal
    Tracks in Lost state for more than track_buffer frames are deleted.

WHY BYTETRACK FOR CROWDED SCENES
=================================
  • In a typical dense-traffic frame, 30–50% of objects are partially occluded.
    Their YOLO scores drop below a naive threshold (e.g. 0.5), causing SORT to
    lose them. ByteTrack recovers these via Stage 2.
  • Result: dramatically fewer ID switches and fewer track fragmentations
    compared to SORT, especially in stop-and-go, pedestrian-crossing, and
    turn scenarios.

CLASS-AWARE MATCHING
====================
  During IoU cost computation, cross-class pairs are penalized (IoU forced to 0).
  This prevents a stationary motorcycle from "stealing" the ID of a nearby car
  when the car is temporarily occluded — a very common failure mode in dense
  aerial footage.

KALMAN FILTER STATE
===================
  Each track maintains a constant-velocity Kalman filter with 8-D state:
      [cx, cy, w, h, vcx, vcy, vw, vh]
  and 4-D measurement [cx, cy, w, h].  Between frames the filter predicts
  where the object *should* be, making the IoU cost more robust to brief gaps.
"""

from __future__ import annotations

import enum
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Track state
# ---------------------------------------------------------------------------

class TrackState(enum.IntEnum):
    New      = 0   # Just initialised, not yet confirmed
    Tracked  = 1   # Receiving regular detections
    Lost     = 2   # Missed for a few frames; still in memory
    Removed  = 3   # Permanently removed


# ---------------------------------------------------------------------------
# IoU utilities
# ---------------------------------------------------------------------------

def iou_matrix(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    """
    Vectorised IoU between two sets of [x1, y1, x2, y2] boxes.

    Returns:
        (N, M) float32 matrix where entry [i, j] = IoU(a_i, b_j).
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = boxes_a[:, np.newaxis, :]   # (N, 1, 4)
    b = boxes_b[np.newaxis, :, :]   # (1, M, 4)

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w   = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h   = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union  = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter_area

    return inter_area / (union + 1e-7)


def hungarian_match(
    detections: List[Dict],
    tracks: List["STrack"],
    distance_threshold: float,
    class_aware: bool = True,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solve the assignment problem and return valid matches.

    Args:
        detections:          List of detection dicts with 'bbox' and 'class_id'.
        tracks:              List of STrack objects.
        distance_threshold:  Maximum *IoU-distance* (= 1 − IoU) to accept a match.
                             0.8 → accepts if IoU ≥ 0.2 (permissive, handles occlusion).
                             0.5 → accepts if IoU ≥ 0.5 (strict, used for Stage 2).
        class_aware:         If True, cross-class pairs are never matched.

    Returns:
        (matches, unmatched_det_indices, unmatched_trk_indices)
    """
    if not detections or not tracks:
        return [], list(range(len(detections))), list(range(len(tracks)))

    det_boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
    trk_boxes = np.array([t.bbox_xyxy for t in tracks],   dtype=np.float32)

    iou_mat = iou_matrix(det_boxes, trk_boxes)

    if class_aware:
        det_cls = np.array([d["class_id"] for d in detections])
        trk_cls = np.array([t.class_id   for t in tracks])
        cross_class = det_cls[:, np.newaxis] != trk_cls[np.newaxis, :]
        iou_mat[cross_class] = 0.0

    cost = 1.0 - iou_mat
    row_ind, col_ind = linear_sum_assignment(cost)

    matches:      List[Tuple[int, int]] = []
    matched_det:  set = set()
    matched_trk:  set = set()

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= distance_threshold:
            matches.append((r, c))
            matched_det.add(r)
            matched_trk.add(c)

    unmatched_dets = [i for i in range(len(detections)) if i not in matched_det]
    unmatched_trks = [i for i in range(len(tracks))     if i not in matched_trk]

    return matches, unmatched_dets, unmatched_trks


# ---------------------------------------------------------------------------
# Single-track Kalman filter
# ---------------------------------------------------------------------------

class STrack:
    """
    A single tracked object with an internal Kalman filter.

    State vector: [cx, cy, w, h, vcx, vcy, vw, vh]
    Measurement:  [cx, cy, w, h]
    Model:        constant-velocity
    """

    _id_counter: int = 0

    # ---- construction --------------------------------------------------------

    def __init__(
        self,
        bbox_xyxy:  List[float],
        score:      float,
        class_id:   int,
        class_name: str,
    ) -> None:
        STrack._id_counter += 1
        self.track_id   = STrack._id_counter
        self.score      = score
        self.class_id   = class_id
        self.class_name = class_name

        self.state          = TrackState.New
        self.is_activated   = False
        self.tracklet_len   = 0
        self.lost_frames    = 0
        self.frame_id       = 0
        self.start_frame    = 0

        self._init_kalman(bbox_xyxy)

    @classmethod
    def reset_id_counter(cls) -> None:
        """Reset the global ID counter (useful between video clips)."""
        cls._id_counter = 0

    # ---- Kalman setup --------------------------------------------------------

    def _init_kalman(self, bbox_xyxy: List[float]) -> None:
        x1, y1, x2, y2 = bbox_xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w  = max(1.0, x2 - x1)
        h  = max(1.0, y2 - y1)

        # 8-state / 4-measurement Kalman filter
        self.kf = cv2.KalmanFilter(8, 4)

        # H: measurement matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)

        # F: constant-velocity transition
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        # Q: process noise (position more uncertain than velocity)
        self.kf.processNoiseCov = np.diag(
            [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 1e-4, 1e-4]
        ).astype(np.float32)

        # R: measurement noise
        self.kf.measurementNoiseCov = np.diag(
            [5.0, 5.0, 5.0, 5.0]
        ).astype(np.float32)

        init = np.array(
            [cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
        ).reshape(8, 1)
        self.kf.statePre  = init.copy()
        self.kf.statePost = init.copy()
        # High initial uncertainty
        self.kf.errorCovPre  = np.eye(8, dtype=np.float32) * 100.0
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 100.0

    # ---- state transitions ---------------------------------------------------

    def predict(self) -> None:
        """Advance Kalman filter by one time step."""
        self.kf.predict()
        if self.state == TrackState.Lost:
            self.lost_frames += 1

    def activate(self, frame_id: int) -> None:
        """Confirm a brand-new track."""
        self.state        = TrackState.Tracked
        self.is_activated = True
        self.frame_id     = frame_id
        self.start_frame  = frame_id
        self.tracklet_len = 1

    def re_activate(
        self,
        bbox_xyxy:  List[float],
        score:      float,
        class_id:   int,
        class_name: str,
        frame_id:   int,
    ) -> None:
        """Recover a previously Lost track."""
        self._kalman_correct(bbox_xyxy)
        self.score      = score
        self.class_id   = class_id
        self.class_name = class_name
        self.state      = TrackState.Tracked
        self.lost_frames = 0
        self.frame_id   = frame_id
        self.tracklet_len += 1

    def update(
        self,
        bbox_xyxy:  List[float],
        score:      float,
        class_id:   int,
        class_name: str,
        frame_id:   int,
    ) -> None:
        """Update an active track with a new matched detection."""
        self._kalman_correct(bbox_xyxy)
        self.score      = score
        self.class_id   = class_id
        self.class_name = class_name
        self.state      = TrackState.Tracked
        self.lost_frames = 0
        self.frame_id   = frame_id
        self.tracklet_len += 1

    def mark_lost(self) -> None:
        """Mark this track as lost (not matched this frame)."""
        self.state       = TrackState.Lost
        self.lost_frames = 0

    # ---- Kalman helper -------------------------------------------------------

    def _kalman_correct(self, bbox_xyxy: List[float]) -> None:
        x1, y1, x2, y2 = bbox_xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w  = max(1.0, x2 - x1)
        h  = max(1.0, y2 - y1)
        meas = np.array([cx, cy, w, h], dtype=np.float32).reshape(4, 1)
        self.kf.correct(meas)

    # ---- state accessors -----------------------------------------------------

    @property
    def bbox_xyxy(self) -> List[float]:
        """Kalman-estimated bounding box in [x1, y1, x2, y2] pixel coords."""
        s = self.kf.statePost.flatten()
        cx, cy, w, h = float(s[0]), float(s[1]), max(1.0, float(s[2])), max(1.0, float(s[3]))
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    @property
    def center(self) -> Tuple[float, float]:
        s = self.kf.statePost.flatten()
        return float(s[0]), float(s[1])

    def __repr__(self) -> str:
        return (f"STrack(id={self.track_id}, cls={self.class_name}, "
                f"state={self.state.name}, len={self.tracklet_len})")


# ---------------------------------------------------------------------------
# ByteTracker
# ---------------------------------------------------------------------------

class BYTETracker:
    """
    ByteTrack multi-object tracker tuned for dense aerial traffic.

    Parameters
    ----------
    high_thresh:    Minimum score for a detection to be used in Stage-1 matching
                    and to initialise new tracks. Default 0.5.
    low_thresh:     Minimum score for a detection to be used in Stage-2 matching.
                    Default 0.1.
    match_thresh:   Maximum IoU-distance to accept a match in Stage 1.
                    0.8 → accepts if IoU ≥ 0.2. Generous to handle occlusion.
    track_buffer:   Frames a Lost track is kept in memory before deletion.
                    30 frames @ 25 fps = 1.2 seconds of re-identification window.
    min_hits:       Consecutive matches before a track appears in output.
                    Avoids showing ephemeral false-positive tracks.
    class_aware:    Penalise cross-class assignments to prevent class switching.
    """

    def __init__(
        self,
        high_thresh:  float = 0.50,
        low_thresh:   float = 0.10,
        match_thresh: float = 0.80,
        track_buffer: int   = 30,
        min_hits:     int   = 3,
        class_aware:  bool  = True,
    ) -> None:
        self.high_thresh  = high_thresh
        self.low_thresh   = low_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_hits     = min_hits
        self.class_aware  = class_aware

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks:    List[STrack] = []
        self.frame_id = 0

        logger.info(
            "BYTETracker — high_thresh=%.2f  low_thresh=%.2f  match_thresh=%.2f  "
            "track_buffer=%d  min_hits=%d  class_aware=%s",
            high_thresh, low_thresh, match_thresh, track_buffer, min_hits, class_aware,
        )

    # ---- public update -------------------------------------------------------

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Process one frame of detections and return active tracks.

        Args:
            detections: Output of Detector.detect() for the current frame.

        Returns:
            List of active track dicts::
                {
                  "track_id":    int,
                  "bbox":        [x1, y1, x2, y2],
                  "center":      (cx, cy),
                  "class_id":    int,
                  "class_name":  str,
                  "confidence":  float,
                  "tracklet_len": int,
                }
        """
        self.frame_id += 1

        # ---- split detections by confidence ---------------------------------
        dets_high = [d for d in detections if d["confidence"] >= self.high_thresh]
        dets_low  = [d for d in detections
                     if self.low_thresh <= d["confidence"] < self.high_thresh]

        # ---- Kalman prediction for all known tracks -------------------------
        for t in self.tracked_stracks + self.lost_stracks:
            t.predict()

        # ---- Stage 1: high-conf dets ↔ all known tracks --------------------
        all_known = list(self.tracked_stracks) + list(self.lost_stracks)

        matched1, unmatched_high, unmatched_all = hungarian_match(
            dets_high, all_known,
            distance_threshold=self.match_thresh,
            class_aware=self.class_aware,
        )

        activated_ids: set = set()

        for d_idx, t_idx in matched1:
            t   = all_known[t_idx]
            det = dets_high[d_idx]
            if t.state == TrackState.Lost:
                t.re_activate(det["bbox"], det["confidence"],
                               det["class_id"], det["class_name"], self.frame_id)
            else:
                t.update(det["bbox"], det["confidence"],
                          det["class_id"], det["class_name"], self.frame_id)
            activated_ids.add(t.track_id)

        # ---- Stage 2: low-conf dets ↔ unmatched *Tracked* tracks ----------
        remaining_tracked = [
            all_known[i]
            for i in unmatched_all
            if (all_known[i].state == TrackState.Tracked
                and all_known[i].track_id not in activated_ids)
        ]

        matched2, _, still_unmatched = hungarian_match(
            dets_low, remaining_tracked,
            distance_threshold=0.50,   # stricter for low-conf associations
            class_aware=self.class_aware,
        )

        for d_idx, t_idx in matched2:
            t   = remaining_tracked[t_idx]
            det = dets_low[d_idx]
            t.update(det["bbox"], det["confidence"],
                      det["class_id"], det["class_name"], self.frame_id)
            activated_ids.add(t.track_id)

        # ---- Mark unmatched Tracked tracks as Lost --------------------------
        for i in still_unmatched:
            remaining_tracked[i].mark_lost()

        # Also mark Tracked tracks from all_known that were never updated
        for t in self.tracked_stracks:
            if t.track_id not in activated_ids and t.state == TrackState.Tracked:
                t.mark_lost()

        # ---- Initialise new tracks from unmatched high-conf dets -----------
        new_stracks: List[STrack] = []
        for d_idx in unmatched_high:
            det = dets_high[d_idx]
            nt  = STrack(det["bbox"], det["confidence"],
                         det["class_id"], det["class_name"])
            nt.activate(self.frame_id)
            new_stracks.append(nt)

        # ---- Rebuild state lists --------------------------------------------
        self.tracked_stracks = (
            [t for t in all_known if t.state == TrackState.Tracked]
            + new_stracks
        )
        # Remove duplicates by track_id (safety guard)
        seen: set = set()
        deduped: List[STrack] = []
        for t in self.tracked_stracks:
            if t.track_id not in seen:
                deduped.append(t)
                seen.add(t.track_id)
        self.tracked_stracks = deduped

        # Keep only non-expired Lost tracks
        self.lost_stracks = [
            t for t in all_known
            if t.state == TrackState.Lost and t.lost_frames < self.track_buffer
        ]
        # Deduplicate lost
        seen = set()
        self.lost_stracks = [
            t for t in self.lost_stracks
            if not (t.track_id in seen or seen.add(t.track_id))   # type: ignore[func-returns-value]
        ]

        # ---- Build output ---------------------------------------------------
        output: List[Dict] = []
        for t in self.tracked_stracks:
            # Suppress brand-new tracks until min_hits are accumulated
            # (except in the very first min_hits frames of the video).
            if t.tracklet_len >= self.min_hits or self.frame_id <= self.min_hits:
                x1, y1, x2, y2 = t.bbox_xyxy
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                output.append({
                    "track_id":    t.track_id,
                    "bbox":        [x1, y1, x2, y2],
                    "center":      (cx, cy),
                    "class_id":    t.class_id,
                    "class_name":  t.class_name,
                    "confidence":  t.score,
                    "tracklet_len": t.tracklet_len,
                })

        logger.debug(
            "Frame %d — tracked=%d  lost=%d  output=%d",
            self.frame_id,
            len(self.tracked_stracks),
            len(self.lost_stracks),
            len(output),
        )
        return output

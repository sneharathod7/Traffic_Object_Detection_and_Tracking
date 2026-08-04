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

# pyrefly: ignore [missing-import]
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from reid import ReIDExtractor, compute_appearance_distance

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


def diou_matrix(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    """
    Vectorised DIoU (Distance IoU) between two sets of [x1, y1, x2, y2] boxes.
    DIoU = IoU - (Distance_Between_Centers^2) / (Diagonal_Of_Enclosing_Box^2)
    This solves bounding box collisions in dense crowds.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = boxes_a[:, np.newaxis, :]   # (N, 1, 4)
    b = boxes_b[np.newaxis, :, :]   # (1, M, 4)

    # IoU
    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w   = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h   = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union  = area_a + area_b - inter_area
    iou = inter_area / (union + 1e-7)

    # Center distances (rho^2)
    cx_a = (a[..., 0] + a[..., 2]) / 2.0
    cy_a = (a[..., 1] + a[..., 3]) / 2.0
    cx_b = (b[..., 0] + b[..., 2]) / 2.0
    cy_b = (b[..., 1] + b[..., 3]) / 2.0
    rho2 = (cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2

    # Smallest enclosing box diagonal (c^2)
    enc_x1 = np.minimum(a[..., 0], b[..., 0])
    enc_y1 = np.minimum(a[..., 1], b[..., 1])
    enc_x2 = np.maximum(a[..., 2], b[..., 2])
    enc_y2 = np.maximum(a[..., 3], b[..., 3])
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7

    diou = iou - (rho2 / c2)
    return diou


def hungarian_match(
    detections: List[Dict],
    tracks: List["STrack"],
    distance_threshold: float,
    class_aware: bool = True,
    motorcycle_match_thresh: float = 0.70,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solve the assignment problem and return valid matches.

    Args:
        detections:          List of detection dicts with 'bbox', 'class_id', and optional 'emb'.
        tracks:              List of STrack objects.
        distance_threshold:  Maximum *IoU-distance* (= 1 − IoU) to accept a match for other classes.
        class_aware:         If True, cross-class pairs are never matched.
        motorcycle_match_thresh: Maximum distance threshold to accept a motorcycle match.

    Returns:
        (matches, unmatched_det_indices, unmatched_trk_indices)
    """
    if not detections or not tracks:
        return [], list(range(len(detections))), list(range(len(tracks)))

    det_boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
    trk_boxes = np.array([t.bbox_xyxy for t in tracks],   dtype=np.float32)

    diou_mat = diou_matrix(det_boxes, trk_boxes)

    det_cls = np.array([d["class_id"] for d in detections])
    trk_cls = np.array([t.class_id   for t in tracks])
    cross_class = det_cls[:, np.newaxis] != trk_cls[np.newaxis, :]

    if class_aware:
        diou_mat[cross_class] = -1.0 # Max distance for cross-class
    else:
        # Soft penalty: reduces effective DIoU by 0.2 for cross-class matches
        diou_mat[cross_class] = np.maximum(-1.0, diou_mat[cross_class] - 0.2)

    # Build the distance/cost matrix. Since DIoU ranges [-1, 1], cost ranges [0, 2].
    cost = 1.0 - diou_mat

    # Apply Motion Gating and Appearance Fusion
    for r in range(len(detections)):
        det = detections[r]
        det_bbox = det["bbox"]
        det_cx = (det_bbox[0] + det_bbox[2]) / 2.0
        det_cy = (det_bbox[1] + det_bbox[3]) / 2.0
        det_cls_id = det["class_id"]

        for c in range(len(tracks)):
            trk = tracks[c]
            if class_aware and det_cls_id != trk.class_id:
                cost[r, c] = 1.0
                continue

            # Kalman predicted center and track bounding box size
            trk_cx, trk_cy = trk.center
            trk_bbox = trk.bbox_xyxy
            trk_w = trk_bbox[2] - trk_bbox[0]
            trk_h = trk_bbox[3] - trk_bbox[1]
            sz = max(trk_w, trk_h)

            d_center = np.hypot(det_cx - trk_cx, det_cy - trk_cy)

            # Class-aware motion gating limit G
            if det_cls_id == 3:  # motorcycle
                G = 4.5 * sz
            else:
                G = 2.5 * sz

            if d_center > G:
                # Gate match out
                cost[r, c] = 1.0
            else:
                # ALL classes appearance fusion
                # Motion consistency cost component
                cost_motion = d_center / G
                
                # Appearance cost component (deep + color hist)
                if "emb" in det and trk.curr_emb is not None:
                    d_app = compute_appearance_distance(det["emb"], trk.curr_emb)
                else:
                    d_app = 0.5  # Neutral default

                # Fused cost formula: 40% DIoU distance, 20% Motion penalty, 40% Appearance distance
                cost[r, c] = 0.4 * (1.0 - diou_mat[r, c]) + 0.2 * cost_motion + 0.4 * d_app

    row_ind, col_ind = linear_sum_assignment(cost)

    matches:      List[Tuple[int, int]] = []
    matched_det:  set = set()
    matched_trk:  set = set()

    for r, c in zip(row_ind, col_ind):
        det_cls_id = detections[r]["class_id"]
        thresh = motorcycle_match_thresh if det_cls_id == 3 else distance_threshold
        if cost[r, c] <= thresh:
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
        self.class_history = {class_id: 1}
        self.class_name_map = {class_id: class_name}

        self.state          = TrackState.New
        self.is_activated   = False
        self.frame_id       = 0
        self.tracklet_len   = 0
        self.start_frame    = 0
        self.lost_frames    = 0
        self.last_reactivated_frame = -1
        
        # OCSORT: Keep track of the last confident observation
        x1, y1, x2, y2 = bbox_xyxy
        self.last_valid_cx = (x1 + x2) / 2.0
        self.last_valid_cy = (y1 + y2) / 2.0

        self.curr_emb: Optional[Dict[str, np.ndarray]] = None

        self._init_kalman(bbox_xyxy)

    @classmethod
    def reset_id_counter(cls, start: int = 0) -> None:
        """Reset the global ID counter (useful between video clips).
        
        Args:
            start: The value to reset the counter to (default 0).
                   Set to the last known track_id to continue numbering
                   from a checkpoint without gaps or duplicates.
        """
        cls._id_counter = start

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

        # Noise matrices will be updated dynamically based on size
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32)
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32)

        init = np.array(
            [cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
        ).reshape(8, 1)
        self.kf.statePre  = init.copy()
        self.kf.statePost = init.copy()
        # High initial uncertainty
        self.kf.errorCovPre  = np.eye(8, dtype=np.float32) * 100.0
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 100.0
        
        self._update_noise_matrices(w, h)

    def _update_noise_matrices(self, w: float, h: float, confidence: float = 1.0) -> None:
        """Update Kalman filter noise matrices dynamically scaled by size, class, and confidence."""
        # Baseline noise scaling factors
        std_weight_position = 1.0 / 20
        std_weight_velocity = 1.0 / 160
        
        # Layer 3: Class-Specific Kalman Physics
        # Determine class if available (can be None during init)
        cls_id = getattr(self, "class_id", None)
        if cls_id == 3:  # Motorcycle (high agility)
            std_weight_position *= 1.5
            std_weight_velocity *= 2.0
        elif cls_id == 5:  # Bus (low agility, extreme momentum)
            std_weight_position *= 0.5
            std_weight_velocity *= 0.2

        sz = max(w, h)
        p_std = std_weight_position * sz
        v_std = std_weight_velocity * sz
        
        # Process noise Q
        q = np.diag([p_std, p_std, p_std, p_std, v_std, v_std, v_std, v_std]) ** 2
        
        # Layer 8: Uncertainty Estimation
        # Lower YOLO confidence = exponentially higher measurement uncertainty
        uncertainty_factor = 1.0 + (1.0 - confidence) * 5.0
        r = np.diag([p_std, p_std, p_std, p_std]) ** 2 * uncertainty_factor
            
        self.kf.processNoiseCov = q.astype(np.float32)
        self.kf.measurementNoiseCov = r.astype(np.float32)

    # ---- appearance update helper --------------------------------------------

    def update_appearance(self, emb: Optional[Dict[str, np.ndarray]], alpha: float = 0.85) -> None:
        """
        Update the appearance embedding via Exponential Moving Average (EMA).
        """
        if emb is None:
            return
        if self.curr_emb is None:
            self.curr_emb = {
                "cnn": emb["cnn"].copy(),
                "color": emb["color"].copy(),
            }
        else:
            # Deep features update
            self.curr_emb["cnn"] = alpha * self.curr_emb["cnn"] + (1.0 - alpha) * emb["cnn"]
            self.curr_emb["cnn"] /= np.linalg.norm(self.curr_emb["cnn"]) + 1e-7
            
            # Color histogram update
            self.curr_emb["color"] = alpha * self.curr_emb["color"] + (1.0 - alpha) * emb["color"]
            self.curr_emb["color"] /= np.linalg.norm(self.curr_emb["color"]) + 1e-7

    # ---- state transitions ---------------------------------------------------

    def predict(self) -> None:
        """Advance Kalman filter by one time step."""
        # Dynamically scale process noise using current state bounds
        s = self.kf.statePost.flatten()
        w, h = max(1.0, float(s[2])), max(1.0, float(s[3]))
        self._update_noise_matrices(w, h)
        
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
        emb:        Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Recover a previously Lost track."""
        # OOS: Observation-Centric Online Smoothing
        # If the track was lost for multiple frames, the Kalman filter has drifted.
        # We mathematically reconstruct the missing velocity vector to snap it back.
        gap = frame_id - self.frame_id
        if gap > 1:
            x1, y1, x2, y2 = bbox_xyxy
            new_cx = (x1 + x2) / 2.0
            new_cy = (y1 + y2) / 2.0
            
            vx = (new_cx - self.last_valid_cx) / gap
            vy = (new_cy - self.last_valid_cy) / gap
            
            self.kf.statePost[0, 0] = new_cx
            self.kf.statePost[1, 0] = new_cy
            self.kf.statePost[4, 0] = vx
            self.kf.statePost[5, 0] = vy
            # Reset error covariance to trust the new observation
            self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 10.0

        self._kalman_correct(bbox_xyxy, score)
        self.score = score
        
        # Update last valid position
        self.last_valid_cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0
        self.last_valid_cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0
        
        # Majority class voting
        self.class_name_map[class_id] = class_name
        self.class_history[class_id] = self.class_history.get(class_id, 0) + 1
        self.class_id = max(self.class_history.items(), key=lambda x: x[1])[0]
        self.class_name = self.class_name_map[self.class_id]

        self.state      = TrackState.Tracked
        self.lost_frames = 0
        self.frame_id   = frame_id
        self.tracklet_len += 1
        self.last_reactivated_frame = frame_id
        if emb is not None:
            self.update_appearance(emb)

    def update(
        self,
        bbox_xyxy:  List[float],
        score:      float,
        class_id:   int,
        class_name: str,
        frame_id:   int,
        emb:        Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Update an active track with a new matched detection."""
        self._kalman_correct(bbox_xyxy, score)
        self.score = score
        
        # Update last valid position
        self.last_valid_cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0
        self.last_valid_cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0
        
        # Majority class voting
        self.class_name_map[class_id] = class_name
        self.class_history[class_id] = self.class_history.get(class_id, 0) + 1
        self.class_id = max(self.class_history.items(), key=lambda x: x[1])[0]
        self.class_name = self.class_name_map[self.class_id]

        self.state      = TrackState.Tracked
        self.lost_frames = 0
        self.frame_id   = frame_id
        self.tracklet_len += 1
        if emb is not None:
            self.update_appearance(emb)

    def mark_lost(self) -> None:
        """Mark this track as lost (not matched this frame)."""
        self.state       = TrackState.Lost
        self.lost_frames = 0

    def get_max_lost_frames(self, base_buffer: int, motorcycle_buffer: int) -> int:
        """Dynamically scale the allowed lost buffer (Layer 5: Adaptive Track Buffer)."""
        # Base class buffer
        buf = motorcycle_buffer if self.class_id == 3 else base_buffer
        
        # Scale up if it's a very long, established track (up to 2x buffer)
        len_factor = min(2.0, max(1.0, self.tracklet_len / 30.0))
        
        # Scale down if it was low confidence when we lost it (down to 0.5x buffer)
        conf_factor = max(0.5, self.score)
        
        # Motorcycles get extra boost if long-tracked (to handle shadow occlusions)
        if self.class_id == 3 and self.tracklet_len > 15:
            len_factor = max(len_factor, 1.5)
            
        return int(buf * len_factor * conf_factor)

    # ---- Kalman helper -------------------------------------------------------

    def _kalman_correct(self, bbox_xyxy: List[float], confidence: float = 1.0) -> None:
        x1, y1, x2, y2 = bbox_xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w  = max(1.0, x2 - x1)
        h  = max(1.0, y2 - y1)
        self._update_noise_matrices(w, h, confidence)
        
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
        class_aware:  bool  = False,
        motorcycle_track_buffer: int = 60,
        motorcycle_match_thresh: float = 0.70,
        device: Optional[str] = None,
    ) -> None:
        self.high_thresh  = high_thresh
        self.low_thresh   = low_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_hits     = min_hits
        self.class_aware  = class_aware

        self.motorcycle_track_buffer = motorcycle_track_buffer
        self.motorcycle_match_thresh = motorcycle_match_thresh

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks:    List[STrack] = []
        self.frame_id = 0

        # Initialize appearance feature extractor
        self.reid_extractor = ReIDExtractor(device=device)

        logger.info(
            "BYTETracker — high_thresh=%.2f  low_thresh=%.2f  match_thresh=%.2f  "
            "track_buffer=%d  min_hits=%d  class_aware=%s  motorcycle_buffer=%d  motorcycle_thresh=%.2f",
            high_thresh, low_thresh, match_thresh, track_buffer, min_hits, class_aware,
            motorcycle_track_buffer, motorcycle_match_thresh,
        )

    # ---- public update -------------------------------------------------------

    def update(self, detections: List[Dict], frame: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Process one frame of detections and return active tracks.

        Args:
            detections: Output of Detector.detect() for the current frame.
            frame: Raw BGR frame image (needed for appearance embeddings).

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

        # ---- Extract appearance embeddings for ALL valid detections -----------
        valid_dets = [d for d in detections if d["confidence"] >= self.low_thresh]
        if valid_dets and frame is not None:
            crops = []
            H_f, W_f = frame.shape[:2]
            for d in valid_dets:
                x1, y1, x2, y2 = d["bbox"]
                rx1 = max(0, int(round(x1)))
                ry1 = max(0, int(round(y1)))
                rx2 = min(W_f, int(round(x2)))
                ry2 = min(H_f, int(round(y2)))
                if rx2 > rx1 and ry2 > ry1:
                    crops.append(frame[ry1:ry2, rx1:rx2].copy())
                else:
                    crops.append(np.zeros((10, 10, 3), dtype=np.uint8))
            
            embs = self.reid_extractor.extract_combined(crops)
            for d, emb in zip(valid_dets, embs):
                d["emb"] = emb

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
            motorcycle_match_thresh=self.motorcycle_match_thresh,
        )

        activated_ids: set = set()

        for d_idx, t_idx in matched1:
            t   = all_known[t_idx]
            det = dets_high[d_idx]
            if t.state == TrackState.Lost:
                t.re_activate(det["bbox"], det["confidence"],
                               det["class_id"], det["class_name"], self.frame_id, det.get("emb"))
            else:
                t.update(det["bbox"], det["confidence"],
                          det["class_id"], det["class_name"], self.frame_id, det.get("emb"))
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
            motorcycle_match_thresh=self.motorcycle_match_thresh,
        )

        for d_idx, t_idx in matched2:
            t   = remaining_tracked[t_idx]
            det = dets_low[d_idx]
            t.update(det["bbox"], det["confidence"],
                      det["class_id"], det["class_name"], self.frame_id, det.get("emb"))
            activated_ids.add(t.track_id)

        # ---- Mark unmatched Tracked tracks as Lost --------------------------
        for i in still_unmatched:
            remaining_tracked[i].mark_lost()

        # Also mark Tracked tracks from all_known that were never updated
        for t in self.tracked_stracks:
            if t.track_id not in activated_ids and t.state == TrackState.Tracked:
                t.mark_lost()

        # ---- Stage 3: OCR (Observation-Centric Recovery) ---------------------
        # Match remaining high-conf detections to Lost tracks using last_valid
        # position (ignoring drifted Kalman filter predictions).
        reconnected_det_indices = []
        for d_idx in unmatched_high:
            det = dets_high[d_idx]
            det_bbox = det["bbox"]
            det_cx = (det_bbox[0] + det_bbox[2]) / 2.0
            det_cy = (det_bbox[1] + det_bbox[3]) / 2.0
            det_w = det_bbox[2] - det_bbox[0]
            det_h = det_bbox[3] - det_bbox[1]
            sz = max(det_w, det_h)

            best_lost_track = None
            best_recon_score = -1.0

            for t in self.lost_stracks:
                curr_gap = self.frame_id - t.frame_id
                if not (2 <= curr_gap <= self.track_buffer):
                    continue

                # Class penalty
                class_penalty = 0.0
                if self.class_aware and det["class_id"] != t.class_id:
                    continue
                elif det["class_id"] != t.class_id:
                    class_penalty = 0.2

                # Size ratio check
                t_bbox = t.bbox_xyxy
                t_w = max(1.0, float(t_bbox[2] - t_bbox[0]))
                t_h = max(1.0, float(t_bbox[3] - t_bbox[1]))
                if not (0.6 <= det_w / t_w <= 1.6) or not (0.6 <= det_h / t_h <= 1.6):
                    continue

                # OCSORT OCR: Spatial distance using last valid observation!
                d_spatial = np.hypot(det_cx - t.last_valid_cx, det_cy - t.last_valid_cy)
                
                # Dynamic search radius based on gap and object size
                max_dist = max(3.0 * sz, 1.5 * sz * np.sqrt(curr_gap))
                if d_spatial > max_dist:
                    continue

                # Appearance similarity check (if available, e.g., motorcycles)
                if "emb" in det and t.curr_emb is not None:
                    d_app = compute_appearance_distance(det["emb"], t.curr_emb)
                    if d_app >= 0.35:
                        continue
                    recon_score = (1.0 - d_app) * (1.0 - d_spatial / max_dist) - class_penalty
                else:
                    # Spatial-only score
                    recon_score = (1.0 - d_spatial / max_dist) - class_penalty

                if recon_score > best_recon_score and recon_score > 0.1:
                    best_recon_score = recon_score
                    best_lost_track = t

            if best_lost_track is not None:
                # Reconnect original ID!
                t = best_lost_track
                t.re_activate(det["bbox"], det["confidence"],
                               det["class_id"], det["class_name"], self.frame_id, det.get("emb"))
                self.tracked_stracks.append(t)
                self.lost_stracks.remove(t)
                reconnected_det_indices.append(d_idx)
                activated_ids.add(t.track_id)
                logger.info(
                    "OCR: Reconnected fragmented track ID %d for %s after gap of %d frames.",
                    t.track_id, t.class_name, self.frame_id - t.frame_id
                )

        unmatched_high = [i for i in unmatched_high if i not in reconnected_det_indices]

        # ---- Initialise new tracks from unmatched high-conf dets -----------
        new_stracks: List[STrack] = []
        for d_idx in unmatched_high:
            det = dets_high[d_idx]
            nt  = STrack(det["bbox"], det["confidence"],
                         det["class_id"], det["class_name"])
            nt.activate(self.frame_id)
            if "emb" in det:
                nt.update_appearance(det["emb"])
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

        # Keep only non-expired Lost tracks (Layer 5: Adaptive track buffer check)
        self.lost_stracks = [
            t for t in all_known
            if t.state == TrackState.Lost and
               t.lost_frames < t.get_max_lost_frames(self.track_buffer, self.motorcycle_track_buffer)
        ]
        # Deduplicate lost
        seen = set()
        self.lost_stracks = [
            t for t in self.lost_stracks
            if not (t.track_id in seen or seen.add(t.track_id))   # type: ignore[func-returns-value]
        ]

        # ---- Fix 1: Suppress Duplicate IDs (IoU-NMS on live tracks) ---------
        self._suppress_duplicate_tracks(iou_threshold=0.65)

        # ---- Fix 2: Repair nascent ID switches before they become permanent --
        self._repair_nascent_id_switches(nascent_window=5, max_recent_loss=3)

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
                    "last_reactivated_frame": t.last_reactivated_frame,
                })

        logger.debug(
            "Frame %d — tracked=%d  lost=%d  output=%d",
            self.frame_id,
            len(self.tracked_stracks),
            len(self.lost_stracks),
            len(output),
        )
        return output

    # ---- Fix 1: Suppress duplicate tracks with heavily-overlapping boxes ----

    def _suppress_duplicate_tracks(self, iou_threshold: float = 0.65) -> None:
        """
        Run a final IoU-NMS pass over all active tracked_stracks.

        Two bounding boxes overlapping by IoU > iou_threshold cannot physically
        belong to two different rigid 3D vehicles — one of them is a ghost
        (typically generated by SAHI slicing the same vehicle across two tiles).

        Resolution:
          - Keep the track with the LONGER tracklet_len (the established track).
          - If tied on tracklet_len, keep the one with HIGHER score.
          - Mark the ghost as Removed immediately; do NOT transfer any counts.
          - The ghost does NOT go to lost_stracks; it is evicted permanently.
        """
        if len(self.tracked_stracks) < 2:
            return

        # Build arrays for vectorised IoU
        boxes = np.array(
            [t.bbox_xyxy for t in self.tracked_stracks], dtype=np.float32
        )
        n = len(boxes)

        # Pairwise IoU  (N x N), lower triangle only to avoid double-processing
        iou_mat = iou_matrix(boxes, boxes)

        suppressed: set = set()  # indices into self.tracked_stracks to kill

        for i in range(n):
            if i in suppressed:
                continue
            for j in range(i + 1, n):
                if j in suppressed:
                    continue
                if iou_mat[i, j] <= iou_threshold:
                    continue

                t_i = self.tracked_stracks[i]
                t_j = self.tracked_stracks[j]

                # Must be the same class — different classes can overlap legitimately
                # (e.g., a person standing next to a motorcycle)
                if t_i.class_id != t_j.class_id:
                    continue

                # Determine winner (longer history) and ghost
                if t_i.tracklet_len > t_j.tracklet_len:
                    ghost_idx = j
                elif t_j.tracklet_len > t_i.tracklet_len:
                    ghost_idx = i
                else:
                    # Tie: keep the higher-confidence track
                    ghost_idx = j if t_i.score >= t_j.score else i

                suppressed.add(ghost_idx)
                ghost = self.tracked_stracks[ghost_idx]
                ghost.state = TrackState.Removed
                logger.debug(
                    "DupNMS: Killed ghost track ID %d (cls=%s, len=%d) overlapping"
                    " with track ID %d (IoU=%.3f, frame=%d).",
                    ghost.track_id, ghost.class_name, ghost.tracklet_len,
                    self.tracked_stracks[i if ghost_idx == j else j].track_id,
                    iou_mat[i, j], self.frame_id,
                )

        # Remove all suppressed ghosts in one pass
        if suppressed:
            self.tracked_stracks = [
                t for idx, t in enumerate(self.tracked_stracks)
                if idx not in suppressed
            ]

    # ---- Fix 2: Retroactively repair nascent ID switches --------------------

    def _repair_nascent_id_switches(self,
                                     nascent_window: int = 5,
                                     max_recent_loss: int = 15) -> None:
        """
        Detect and repair ID switches that happen in the first few frames of a
        new track (nascent_window) caused by brief 1-3 frame detection jitter.

        A jitter-induced switch looks like:
          frame N:   existing track T goes Lost (missed detection by 1 frame)
          frame N+1: same vehicle re-detected, gets brand new ID N

        We catch this by scanning every newly-born track N against recently-lost
        tracks T and checking three criteria simultaneously:
          1. Same class_id
          2. Spatial proximity:  dist(centers) <= 2.0 * max(N.w, N.h)
          3. Appearance:         d_app(N.emb, T.emb) < 0.30
          4. T has been lost for at most max_recent_loss frames

        When a match is found, we transfer T's identity into N:
          - N inherits T's track_id, class_history, class_name_map,
            tracklet_len, last_reactivated_frame, and appearance embedding.
          - T is evicted from lost_stracks immediately.
          - The global _id_counter is NOT decremented (the ghost ID is simply
            abandoned — IDs are cheap, correctness is not).
        """
        if not self.lost_stracks:
            return

        repairs: List[Tuple[STrack, STrack]] = []  # (nascent, lost_donor)
        claimed_lost_ids: set = set()  # prevent one lost track fixing two nascents

        for nascent in self.tracked_stracks:
            # Only consider tracks fresh enough to have been a switch
            if nascent.tracklet_len > nascent_window:
                continue

            n_bbox = nascent.bbox_xyxy
            n_cx = (n_bbox[0] + n_bbox[2]) / 2.0
            n_cy = (n_bbox[1] + n_bbox[3]) / 2.0
            n_w  = n_bbox[2] - n_bbox[0]
            n_h  = n_bbox[3] - n_bbox[1]
            spatial_gate = 2.0 * max(n_w, n_h)

            best_donor: Optional[STrack] = None
            best_score: float = -1.0

            for lost in self.lost_stracks:
                # Criterion 1: class must match exactly
                if lost.class_id != nascent.class_id:
                    continue

                # Criterion 2: must have gone lost very recently
                if lost.lost_frames > max_recent_loss:
                    continue

                # Criterion 3: cannot already be claimed by another nascent
                if lost.track_id in claimed_lost_ids:
                    continue

                # Criterion 4: spatial proximity using last_valid observation
                d_spatial = np.hypot(
                    n_cx - lost.last_valid_cx,
                    n_cy - lost.last_valid_cy,
                )
                if d_spatial > spatial_gate:
                    continue

                # Criterion 5: appearance similarity (if both have embeddings)
                if nascent.curr_emb is not None and lost.curr_emb is not None:
                    d_app = compute_appearance_distance(nascent.curr_emb, lost.curr_emb)
                    if d_app >= 0.30:
                        continue
                    # Composite repair score: blend spatial closeness + visual similarity
                    repair_score = (
                        0.5 * (1.0 - d_app) +
                        0.5 * (1.0 - d_spatial / spatial_gate)
                    )
                else:
                    # Spatial-only score when embeddings unavailable
                    repair_score = 1.0 - d_spatial / spatial_gate

                if repair_score > best_score:
                    best_score = repair_score
                    best_donor = lost

            if best_donor is not None:
                repairs.append((nascent, best_donor))
                claimed_lost_ids.add(best_donor.track_id)

        # Apply repairs: transfer identity from lost donor → nascent track
        lost_ids_to_evict: set = set()
        for nascent, donor in repairs:
            logger.info(
                "IDRepair: Replaced nascent ID %d (len=%d) with recovered ID %d"
                " (cls=%s, donor_lost_frames=%d, frame=%d).",
                nascent.track_id, nascent.tracklet_len, donor.track_id,
                donor.class_name, donor.lost_frames, self.frame_id,
            )
            # Transfer identity — the nascent track keeps its Kalman state
            # (it has the fresh, accurate observation) but gets the historical
            # identity of the donor so the output ID is consistent.
            nascent.track_id          = donor.track_id
            nascent.class_history     = {**donor.class_history}   # deep copy
            nascent.class_name_map    = {**donor.class_name_map}  # deep copy
            nascent.tracklet_len      = donor.tracklet_len + nascent.tracklet_len
            nascent.start_frame       = donor.start_frame
            nascent.last_reactivated_frame = donor.last_reactivated_frame
            nascent.last_valid_cx     = donor.last_valid_cx
            nascent.last_valid_cy     = donor.last_valid_cy

            # Merge appearance: keep the donor's long-term EMA, nudge with nascent's fresh emb
            if donor.curr_emb is not None:
                nascent.curr_emb = {
                    "cnn":   donor.curr_emb["cnn"].copy(),
                    "color": donor.curr_emb["color"].copy(),
                }
                # Now apply nascent's fresh observation into the merged embedding
                if nascent.curr_emb is not None:
                    nascent.update_appearance(nascent.curr_emb, alpha=0.85)

            # Re-resolve class via majority vote after merge
            nascent.class_id   = max(nascent.class_history.items(), key=lambda x: x[1])[0]
            nascent.class_name = nascent.class_name_map.get(nascent.class_id, nascent.class_name)

            lost_ids_to_evict.add(donor.track_id)

        # Evict all donors from lost_stracks in one pass
        if lost_ids_to_evict:
            self.lost_stracks = [
                t for t in self.lost_stracks
                if t.track_id not in lost_ids_to_evict
            ]

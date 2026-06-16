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
import json
import logging
import math
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# pyrefly: ignore [missing-import]
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from reid import ReIDExtractor, compute_appearance_distance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default association config (overridden by config.yaml when loaded)
# ---------------------------------------------------------------------------

DEFAULT_ASSOCIATION_CONFIG: Dict = {
    "association": {
        "weights": {
            "iou": 0.30,
            "appearance": 0.20,
            "direction": 0.20,
            "trajectory": 0.15,
            "motion": 0.10,
            "scale": 0.05,
        },
        "gate_motorcycle": 4.5,
        "gate_default": 2.5,
        "direction_min_history": 5,
    },
    "trajectory": {
        "position_buffer_size": 30,
        "velocity_buffer_size": 10,
        "heading_buffer_size": 10,
        "heading_smoothing_window": 5,
    },
    "recovery": {
        "use_extrapolation": True,
        "extrapolation_max_gap": 30,
        "extrapolation_radius_factor": 1.5,
        "memory_tau_car": 20.0,
        "memory_tau_motorcycle": 25.0,
        "memory_tau_bus": 15.0,
        "memory_tau_person": 15.0,
        "memory_tau_default": 20.0,
    },
    "reliability": {
        "age_weight": 0.20,
        "appearance_weight": 0.30,
        "trajectory_weight": 0.30,
        "association_weight": 0.20,
        "min_reliable": 0.40,
    },
    "switch_logging": {
        "enabled": True,
        "output_path": "outputs/metrics/switch_log.json",
    },
}


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


def _compute_direction_cost(det_cx: float, det_cy: float,
                            trk: "STrack",
                            min_history: int = 5) -> float:
    """
    Compute heading consistency cost between a detection and a track.
    Returns 0.0 (perfect match) to 1.0 (opposite directions).
    Uses multi-frame heading from the track's heading_buffer.
    """
    if len(trk.trajectory_buffer) < min_history:
        return 0.5  # Neutral — not enough history to judge

    # Detection's implied heading relative to track's last position
    last_cx, last_cy, _ = trk.trajectory_buffer[-1]
    dx = det_cx - last_cx
    dy = det_cy - last_cy
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return 0.0  # Essentially stationary — any direction is fine

    det_heading = math.atan2(dy, dx)
    trk_heading = trk.avg_heading

    # Angular difference in [0, pi]
    diff = abs(det_heading - trk_heading)
    if diff > math.pi:
        diff = 2 * math.pi - diff

    # Normalize to [0, 1]
    return diff / math.pi


def _compute_trajectory_cost(det_cx: float, det_cy: float,
                             trk: "STrack") -> float:
    """
    How well does the detection align with the track's extrapolated trajectory?
    Returns 0.0 (exactly on extrapolated path) to 1.0 (far off path).
    """
    if len(trk.trajectory_buffer) < 3:
        return 0.5  # Neutral

    # Extrapolate from last position using average velocity
    last_cx, last_cy, last_frame = trk.trajectory_buffer[-1]
    vx, vy = trk.avg_velocity
    # Assume detection is 1 frame ahead
    pred_cx = last_cx + vx
    pred_cy = last_cy + vy

    d = math.hypot(det_cx - pred_cx, det_cy - pred_cy)

    # Normalize by track bounding box size
    bbox = trk.bbox_xyxy
    sz = max(1.0, max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    return min(1.0, d / (2.0 * sz))


def _compute_scale_cost(det_bbox: List[float], trk: "STrack") -> float:
    """
    Bounding box area ratio cost. Returns 0.0 (same size) to 1.0 (very different).
    """
    det_area = max(1.0, (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1]))
    trk_area = max(1.0, trk.bbox_area_ema) if trk.bbox_area_ema > 0 else det_area
    ratio = max(det_area, trk_area) / min(det_area, trk_area)
    # ratio >= 1.0; map to [0, 1] with soft saturation
    return min(1.0, (ratio - 1.0) / 2.0)


def hungarian_match(
    detections: List[Dict],
    tracks: List["STrack"],
    distance_threshold: float,
    class_aware: bool = True,
    motorcycle_match_thresh: float = 0.70,
    config: Optional[Dict] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    6-component assignment: DIoU + Motion + Appearance + Direction + Trajectory + Scale.

    Args:
        detections:          List of detection dicts with 'bbox', 'class_id', optional 'emb'.
        tracks:              List of STrack objects.
        distance_threshold:  Max cost to accept a match (non-motorcycle).
        class_aware:         If True, cross-class pairs are never matched.
        motorcycle_match_thresh: Max cost for motorcycle matches.
        config:              Association config dict (from config.yaml).

    Returns:
        (matches, unmatched_det_indices, unmatched_trk_indices)
    """
    if not detections or not tracks:
        return [], list(range(len(detections))), list(range(len(tracks)))

    cfg = (config or DEFAULT_ASSOCIATION_CONFIG).get("association", {})
    weights = cfg.get("weights", DEFAULT_ASSOCIATION_CONFIG["association"]["weights"])
    w_iou = weights.get("iou", 0.30)
    w_app = weights.get("appearance", 0.20)
    w_dir = weights.get("direction", 0.20)
    w_traj = weights.get("trajectory", 0.15)
    w_mot = weights.get("motion", 0.10)
    w_scl = weights.get("scale", 0.05)
    G_moto = cfg.get("gate_motorcycle", 4.5)
    G_default = cfg.get("gate_default", 2.5)
    dir_min_hist = cfg.get("direction_min_history", 5)

    det_boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
    trk_boxes = np.array([t.bbox_xyxy for t in tracks],   dtype=np.float32)

    diou_mat = diou_matrix(det_boxes, trk_boxes)

    det_cls = np.array([d["class_id"] for d in detections])
    trk_cls = np.array([t.class_id   for t in tracks])
    cross_class = det_cls[:, np.newaxis] != trk_cls[np.newaxis, :]

    if class_aware:
        diou_mat[cross_class] = -1.0
    else:
        diou_mat[cross_class] = np.maximum(-1.0, diou_mat[cross_class] - 0.2)

    cost = np.ones((len(detections), len(tracks)), dtype=np.float64)

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

            trk_cx, trk_cy = trk.center
            trk_bbox = trk.bbox_xyxy
            trk_w = trk_bbox[2] - trk_bbox[0]
            trk_h = trk_bbox[3] - trk_bbox[1]
            sz = max(trk_w, trk_h)

            d_center = np.hypot(det_cx - trk_cx, det_cy - trk_cy)

            G = G_moto * sz if det_cls_id == 3 else G_default * sz

            if d_center > G:
                cost[r, c] = 1.0
                continue

            # Component 1: DIoU
            c_iou = 1.0 - diou_mat[r, c]  # [0, 2] range, clamp to [0, 1]
            c_iou = min(1.0, max(0.0, c_iou))

            # Component 2: Motion distance
            c_mot = d_center / G

            # Component 3: Appearance
            if "emb" in det and trk.curr_emb is not None:
                c_app = compute_appearance_distance(det["emb"], trk.curr_emb)
            else:
                c_app = 0.5

            # Component 4: Direction consistency
            c_dir = _compute_direction_cost(det_cx, det_cy, trk, dir_min_hist)

            # Component 5: Trajectory consistency
            c_traj = _compute_trajectory_cost(det_cx, det_cy, trk)

            # Component 6: Scale consistency
            c_scl = _compute_scale_cost(det_bbox, trk)

            # Fused 6-component cost
            cost[r, c] = (
                w_iou  * c_iou +
                w_mot  * c_mot +
                w_app  * c_app +
                w_dir  * c_dir +
                w_traj * c_traj +
                w_scl  * c_scl
            )

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
        config:     Optional[Dict] = None,
    ) -> None:
        STrack._id_counter += 1
        self.track_id   = STrack._id_counter
        self.original_id = self.track_id
        self.resurrection_parent: Optional[int] = None
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

        # --- Stage 1: Trajectory Memory ---
        traj_cfg = (config or DEFAULT_ASSOCIATION_CONFIG).get("trajectory", {})
        pos_buf = traj_cfg.get("position_buffer_size", 30)
        vel_buf = traj_cfg.get("velocity_buffer_size", 10)
        hdg_buf = traj_cfg.get("heading_buffer_size", 10)

        self.trajectory_buffer: deque = deque(maxlen=pos_buf)
        self.velocity_buffer: deque   = deque(maxlen=vel_buf)
        self.heading_buffer: deque    = deque(maxlen=hdg_buf)
        self.bbox_area_ema: float     = max(1.0, (x2 - x1) * (y2 - y1))

        # Computed trajectory properties (cached per frame)
        self.avg_velocity: Tuple[float, float] = (0.0, 0.0)
        self.avg_heading: float = 0.0

        # --- Stage 4: Reliability score ---
        self.reliability_score: float = 0.1  # starts low, grows with evidence
        self._appearance_consistency: float = 1.0  # running consistency
        self._association_streak: int = 0  # consecutive matched frames

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

    def _update_trajectory(self, cx: float, cy: float, frame_id: int) -> None:
        """Push a new observation into trajectory/velocity/heading buffers."""
        if self.trajectory_buffer:
            prev_cx, prev_cy, prev_frame = self.trajectory_buffer[-1]
            dt = max(1, frame_id - prev_frame)
            vx = (cx - prev_cx) / dt
            vy = (cy - prev_cy) / dt
            self.velocity_buffer.append((vx, vy))

            heading = math.atan2(vy, vx)
            self.heading_buffer.append(heading)

        self.trajectory_buffer.append((cx, cy, frame_id))

        # Update cached averages
        if self.velocity_buffer:
            vels = np.array(self.velocity_buffer)
            self.avg_velocity = (float(np.mean(vels[:, 0])),
                                 float(np.mean(vels[:, 1])))
        if self.heading_buffer:
            # Circular mean for headings
            sins = sum(math.sin(h) for h in self.heading_buffer)
            coss = sum(math.cos(h) for h in self.heading_buffer)
            self.avg_heading = math.atan2(sins, coss)

    def _update_reliability(self, confidence: float) -> None:
        """Update the reliability score based on accumulated evidence."""
        rel_cfg = DEFAULT_ASSOCIATION_CONFIG.get("reliability", {})
        w_age = rel_cfg.get("age_weight", 0.20)
        w_app = rel_cfg.get("appearance_weight", 0.30)
        w_traj = rel_cfg.get("trajectory_weight", 0.30)
        w_assoc = rel_cfg.get("association_weight", 0.20)

        # Age component: saturates at ~60 frames
        age_score = min(1.0, self.tracklet_len / 60.0)

        # Appearance consistency: tracks with stable appearance are reliable
        app_score = self._appearance_consistency

        # Trajectory smoothness: low velocity variance = smooth = reliable
        if len(self.velocity_buffer) >= 3:
            vels = np.array(self.velocity_buffer)
            speed_std = float(np.std(np.linalg.norm(vels, axis=1)))
            traj_score = max(0.0, 1.0 - speed_std / 5.0)
        else:
            traj_score = 0.5

        # Association stability: consecutive frames matched
        assoc_score = min(1.0, self._association_streak / 10.0)

        self.reliability_score = (
            w_age * age_score +
            w_app * app_score +
            w_traj * traj_score +
            w_assoc * assoc_score
        )

    def get_memory_score(self) -> float:
        """Stage 4: Confidence-decaying memory score for lost tracks."""
        rec_cfg = DEFAULT_ASSOCIATION_CONFIG.get("recovery", {})
        cls_taus = {
            2: rec_cfg.get("memory_tau_car", 20.0),
            3: rec_cfg.get("memory_tau_motorcycle", 25.0),
            5: rec_cfg.get("memory_tau_bus", 15.0),
            0: rec_cfg.get("memory_tau_person", 15.0),
        }
        tau = cls_taus.get(self.class_id, rec_cfg.get("memory_tau_default", 20.0))
        # Scale tau by reliability — reliable tracks decay slower
        tau = tau * (0.5 + self.reliability_score)
        return math.exp(-self.lost_frames / max(1.0, tau))

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
        cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0
        cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0
        self.last_valid_cx = cx
        self.last_valid_cy = cy

        # Update trajectory memory
        self._update_trajectory(cx, cy, frame_id)
        self.bbox_area_ema = 0.9 * self.bbox_area_ema + 0.1 * max(1.0, (bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1]))

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
        self._association_streak += 1
        self._update_reliability(score)
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
        cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0
        cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0
        self.last_valid_cx = cx
        self.last_valid_cy = cy

        # Update trajectory memory
        self._update_trajectory(cx, cy, frame_id)
        self.bbox_area_ema = 0.9 * self.bbox_area_ema + 0.1 * max(1.0, (bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1]))

        # Majority class voting
        self.class_name_map[class_id] = class_name
        self.class_history[class_id] = self.class_history.get(class_id, 0) + 1
        self.class_id = max(self.class_history.items(), key=lambda x: x[1])[0]
        self.class_name = self.class_name_map[self.class_id]

        self.state      = TrackState.Tracked
        self.lost_frames = 0
        self.frame_id   = frame_id
        self.tracklet_len += 1
        self._association_streak += 1
        self._update_reliability(score)
        if emb is not None:
            self.update_appearance(emb)

    def mark_lost(self) -> None:
        """Mark this track as lost (not matched this frame)."""
        self.state       = TrackState.Lost
        self.lost_frames = 0
        self._association_streak = 0  # Reset streak on loss

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
        config: Optional[Dict] = None,
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

        # Configuration
        self.config = config or DEFAULT_ASSOCIATION_CONFIG

        # Stage 5: Switch logging
        self._switch_log: List[Dict] = []
        sw_cfg = self.config.get("switch_logging", {})
        self._switch_logging_enabled = sw_cfg.get("enabled", True)
        self._switch_log_path = sw_cfg.get("output_path", "outputs/metrics/switch_log.json")

        # Stage 6: Resurrection parameters
        res_cfg = self.config.get("resurrection", {})
        self.resurrection_enabled = res_cfg.get("enabled", True)
        self.resurrection_max_gap = res_cfg.get("max_gap", 20)
        self.resurrection_score_threshold = res_cfg.get("score_threshold", 0.85)
        self.resurrection_log_threshold = res_cfg.get("log_threshold", 0.70)
        self.resurrection_motorcycle_app = res_cfg.get("motorcycle_app_threshold", 0.90)
        self.resurrection_weights = res_cfg.get("weights", {"appearance": 0.45, "trajectory": 0.30, "heading": 0.15, "memory": 0.10})
        log_cfg = res_cfg.get("logging", {})
        self.resurrection_logging_enabled = log_cfg.get("enabled", True)
        self.resurrection_log_path = log_cfg.get("output_path", "outputs/metrics/resurrection_log.json")
        self._resurrection_log: List[Dict] = []

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
            config=self.config,
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
                if not (2 <= curr_gap <= 30):
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

                # Stage 3: Velocity extrapolation — predict expected position
                rec_cfg = self.config.get("recovery", {})
                use_extrap = rec_cfg.get("use_extrapolation", True)
                extrap_factor = rec_cfg.get("extrapolation_radius_factor", 1.5)

                if use_extrap and t.avg_velocity != (0.0, 0.0):
                    pred_cx = t.last_valid_cx + t.avg_velocity[0] * curr_gap
                    pred_cy = t.last_valid_cy + t.avg_velocity[1] * curr_gap
                else:
                    pred_cx = t.last_valid_cx
                    pred_cy = t.last_valid_cy

                d_spatial = np.hypot(det_cx - pred_cx, det_cy - pred_cy)

                # Dynamic search radius
                max_dist = max(3.0 * sz, extrap_factor * sz * np.sqrt(curr_gap))
                if d_spatial > max_dist:
                    continue

                # Stage 4: Memory score for confidence-decaying track memory
                mem_score = t.get_memory_score()

                # Appearance similarity check
                if "emb" in det and t.curr_emb is not None:
                    d_app = compute_appearance_distance(det["emb"], t.curr_emb)
                    if d_app >= 0.35:
                        continue
                    recon_score = mem_score * (1.0 - d_app) * (1.0 - d_spatial / max_dist) - class_penalty
                else:
                    recon_score = mem_score * (1.0 - d_spatial / max_dist) - class_penalty

                # Direction consistency bonus/penalty for OCR
                if len(t.heading_buffer) >= 3:
                    dir_cost = _compute_direction_cost(det_cx, det_cy, t, 3)
                    if dir_cost > 0.7:  # Nearly opposite direction — reject
                        continue
                    recon_score *= (1.0 - 0.3 * dir_cost)  # Small direction bonus

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

        # ---- Stage 5: Detect potential ID switches -------------------------
        if self._switch_logging_enabled:
            self._detect_switches(dets_high, unmatched_high, all_known)

        # ---- Initialise new tracks from unmatched high-conf dets -----------
        new_stracks: List[STrack] = []
        for d_idx in unmatched_high:
            det = dets_high[d_idx]
            nt  = STrack(det["bbox"], det["confidence"],
                         det["class_id"], det["class_name"],
                         config=self.config)
            nt.activate(self.frame_id)
            if "emb" in det:
                nt.update_appearance(det["emb"])
            new_stracks.append(nt)

        # ---- Stage 6: Track Resurrection Layer ------------------------------
        self._attempt_resurrection(new_stracks)

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



    # ---- Stage 5: Switch detection and logging --------------------------------

    def _detect_switches(
        self,
        dets_high: List[Dict],
        unmatched_high: List[int],
        all_known: List[STrack],
    ) -> None:
        """
        Detect potential ID switches by looking for new tracks being created
        near recently-lost tracks of the same class. Log detailed context for
        every suspected switch.

        This is a DIAGNOSTIC tool — it does not alter tracking behavior.
        It provides the empirical evidence needed to decide whether to invest
        in ReID retraining or tracklet stitching.
        """
        for d_idx in unmatched_high:
            det = dets_high[d_idx]
            det_bbox = det["bbox"]
            det_cx = (det_bbox[0] + det_bbox[2]) / 2.0
            det_cy = (det_bbox[1] + det_bbox[3]) / 2.0
            det_cls = det["class_id"]
            det_w = det_bbox[2] - det_bbox[0]
            det_h = det_bbox[3] - det_bbox[1]
            det_sz = max(det_w, det_h)

            # Check if this new detection is suspiciously close to a recently-lost track
            for lost_t in self.lost_stracks:
                if lost_t.class_id != det_cls:
                    continue
                if lost_t.lost_frames > 10:
                    continue

                d_spatial = np.hypot(det_cx - lost_t.last_valid_cx,
                                     det_cy - lost_t.last_valid_cy)

                if d_spatial > 3.0 * det_sz:
                    continue

                # This looks like a potential ID switch
                d_app = -1.0
                if "emb" in det and lost_t.curr_emb is not None:
                    d_app = compute_appearance_distance(det["emb"], lost_t.curr_emb)

                # Compute IoU with the lost track's last bbox
                lost_bbox = lost_t.bbox_xyxy
                iou_val = float(iou_matrix(
                    np.array([det_bbox], dtype=np.float32),
                    np.array([lost_bbox], dtype=np.float32),
                )[0, 0])

                # Trajectory distance
                if lost_t.avg_velocity != (0.0, 0.0):
                    pred_cx = lost_t.last_valid_cx + lost_t.avg_velocity[0] * lost_t.lost_frames
                    pred_cy = lost_t.last_valid_cy + lost_t.avg_velocity[1] * lost_t.lost_frames
                    d_traj = np.hypot(det_cx - pred_cx, det_cy - pred_cy)
                else:
                    d_traj = d_spatial

                switch_event = {
                    "frame": self.frame_id,
                    "old_id": lost_t.track_id,
                    "new_id": -1,  # will be assigned when new track is created
                    "class": lost_t.class_name,
                    "class_id": det_cls,
                    "lost_frames": lost_t.lost_frames,
                    "spatial_distance": round(float(d_spatial), 2),
                    "trajectory_distance": round(float(d_traj), 2),
                    "appearance_distance": round(float(d_app), 4) if d_app >= 0 else "unavailable",
                    "iou_with_lost": round(float(iou_val), 4),
                    "lost_track_reliability": round(lost_t.reliability_score, 4),
                    "lost_track_age": lost_t.tracklet_len,
                    "lost_track_memory_score": round(lost_t.get_memory_score(), 4),
                    "detection_confidence": round(det["confidence"], 4),
                    "recovery_context": "unmatched_high_near_lost",
                }
                self._switch_log.append(switch_event)
                logger.debug(
                    "SwitchLog: Potential switch — old_id=%d cls=%s d_spatial=%.1f "
                    "d_app=%.3f iou=%.3f lost=%d frame=%d",
                    lost_t.track_id, lost_t.class_name, d_spatial,
                    d_app, iou_val, lost_t.lost_frames, self.frame_id,
                )
                break  # Log only the best candidate per detection

    def flush_switch_log(self) -> None:
        """Write the accumulated switch log to disk as JSON."""
        if not self._switch_log:
            logger.info("Switch log is empty — no potential ID switches detected.")
            return

        out_path = Path(self._switch_log_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_switch_events": len(self._switch_log),
                "events": self._switch_log,
            }, f, indent=2)

        logger.info(
            "Switch log written: %d events → %s",
            len(self._switch_log), out_path,
        )

    def _attempt_resurrection(self, new_stracks: List[STrack]) -> None:
        """
        Stage 6: Track Resurrection Layer
        A surgical identity recovery mechanism that resurrects lost tracks using appearance,
        trajectory, heading, and memory decay scores, preventing ID fragmentation.
        """
        if not getattr(self, "resurrection_enabled", False):
            return

        resurrected_ids = set()

        for nt in new_stracks:
            if getattr(nt, "curr_emb", None) is None:
                continue

            best_score = -1.0
            best_candidate = None
            best_log_entry = None

            for lt in self.lost_stracks:
                if lt.track_id in resurrected_ids:
                    continue

                gap = self.frame_id - lt.frame_id
                if gap < 2 or gap > self.resurrection_max_gap:
                    continue

                # Appearance Score
                if lt.curr_emb is None:
                    continue
                d_app = compute_appearance_distance(nt.curr_emb, lt.curr_emb)
                app_score = 1.0 - d_app

                # Motorcycle Safeguard
                if lt.class_id == 3 and app_score <= self.resurrection_motorcycle_app:
                    continue

                # Trajectory Score
                if lt.avg_velocity != (0.0, 0.0):
                    pred_cx = lt.last_valid_cx + lt.avg_velocity[0] * gap
                    pred_cy = lt.last_valid_cy + lt.avg_velocity[1] * gap
                else:
                    pred_cx = lt.last_valid_cx
                    pred_cy = lt.last_valid_cy

                nt_cx = (nt.bbox_xyxy[0] + nt.bbox_xyxy[2]) / 2.0
                nt_cy = (nt.bbox_xyxy[1] + nt.bbox_xyxy[3]) / 2.0

                d_spatial = math.hypot(nt_cx - pred_cx, nt_cy - pred_cy)
                sz = max(nt.bbox_xyxy[2] - nt.bbox_xyxy[0], nt.bbox_xyxy[3] - nt.bbox_xyxy[1])
                max_dist = max(3.0 * sz, 1.5 * sz * math.sqrt(gap))
                if d_spatial > max_dist:
                    continue
                traj_score = max(0.0, 1.0 - (d_spatial / max_dist))

                # Heading Score
                # cos(delta_heading) using bridge vector since nt is too new to have heading history
                dx = nt_cx - lt.last_valid_cx
                dy = nt_cy - lt.last_valid_cy
                if abs(dx) < 0.5 and abs(dy) < 0.5:
                    hdg_score = 1.0
                else:
                    heading_bridge = math.atan2(dy, dx)
                    heading_old = getattr(lt, "avg_heading", 0.0)
                    delta_heading = heading_old - heading_bridge
                    hdg_score = max(0.0, math.cos(delta_heading))

                # Require cos > 0.8 safeguard
                if hdg_score < 0.8:
                    continue

                # Memory Score
                mem_score = lt.get_memory_score()

                # Final Score
                w = self.resurrection_weights
                final_score = (
                    w.get("appearance", 0.45) * app_score +
                    w.get("trajectory", 0.30) * traj_score +
                    w.get("heading", 0.15) * hdg_score +
                    w.get("memory", 0.10) * mem_score
                )

                log_entry = {
                    "old_track_id": int(lt.track_id),
                    "candidate_track_id": int(nt.track_id),
                    "class": lt.class_name,
                    "gap": int(gap),
                    "appearance_score": float(round(app_score, 4)),
                    "trajectory_score": float(round(traj_score, 4)),
                    "heading_score": float(round(hdg_score, 4)),
                    "memory_score": float(round(mem_score, 4)),
                    "final_score": float(round(final_score, 4)),
                    "decision": "rejected"
                }

                if final_score > best_score:
                    best_score = final_score
                    best_candidate = lt
                    best_log_entry = log_entry

            if best_candidate is not None and best_log_entry is not None:
                if best_score > self.resurrection_score_threshold:
                    best_log_entry["decision"] = "resurrected"
                    
                    if getattr(self, "resurrection_logging_enabled", False):
                        self._resurrection_log.append(best_log_entry)
                        
                    # Transfer ID and Lineage
                    nt.resurrection_parent = best_candidate.track_id
                    nt.track_id = best_candidate.track_id
                    nt.class_history = getattr(best_candidate, "class_history", {}).copy()
                    
                    # Transfer Memory
                    nt.trajectory_buffer = getattr(best_candidate, "trajectory_buffer", deque()).copy()
                    nt.velocity_buffer = getattr(best_candidate, "velocity_buffer", deque()).copy()
                    nt.heading_buffer = getattr(best_candidate, "heading_buffer", deque()).copy()
                    
                    # Ensure old track gets removed from memory correctly
                    best_candidate.state = TrackState.Removed
                    resurrected_ids.add(best_candidate.track_id)
                    
                    logger.info("RESURRECTION: Re-assigned New Track %d to Lost Track %d (Score: %.3f)", 
                                nt.original_id, nt.track_id, best_score)
                                
                elif best_score > self.resurrection_log_threshold:
                    if getattr(self, "resurrection_logging_enabled", False):
                        self._resurrection_log.append(best_log_entry)

    def flush_resurrection_log(self) -> None:
        """Write the accumulated resurrection log to disk as JSON."""
        if not getattr(self, "resurrection_logging_enabled", False) or not hasattr(self, "_resurrection_log"):
            return
            
        if not self._resurrection_log:
            logger.info("Resurrection log is empty.")
            return

        out_path = Path(self.resurrection_log_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_resurrection_attempts": len(self._resurrection_log),
                "events": self._resurrection_log,
            }, f, indent=2)

        logger.info(
            "Resurrection log written: %d events -> %s",
            len(self._resurrection_log), out_path,
        )


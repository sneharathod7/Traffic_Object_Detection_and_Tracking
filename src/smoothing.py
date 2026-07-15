"""
Trajectory Smoothing Module
============================

WHY SMOOTHING IS CRITICAL FOR SAFETY ANALYSIS
===============================================
Even after Kalman-filter based tracking, the *output* coordinates still carry
frame-to-frame jitter caused by:
  • Detection bounding-box instability (YOLO's output varies slightly per frame
    even for a stationary object).
  • Quantisation noise from integer pixel rounding.
  • Brief partial occlusions that shift the box by a few pixels.
  • SAHI tile-boundary artefacts where the same object is detected from
    two adjacent tiles with a small centre-of-mass offset.

For safety-rule logic, jitter translates directly into:
  1. Spurious velocity spikes — an object that is actually stationary may appear
     to jump 5 pixels between frames, which at a scale of 0.05 m/px × 25 fps
     gives a phantom velocity of 6.25 m/s (~22 km/h).
  2. Erratic trajectory curves — the direction-of-travel vector becomes noisy,
     making lane-change or wrong-way detection unreliable.
  3. Incorrect TTC estimates — Time-To-Collision computed from jittery positions
     produces dangerously wide confidence intervals.

THREE IMPLEMENTATIONS (in order of recommended preference)
============================================================

AdaptiveSplineSmoother  ← DEFAULT — best for dense/crowded scenes
--------------------------------------------------------------------
A two-phase hybrid approach:

  Phase 1 (online):
    An Adaptive Exponential Moving Average (EMA) produces a real-time smoothed
    position each frame. The EMA alpha (responsiveness) is dynamically scaled:
      • LOWER alpha (more smoothing) when crowding density is HIGH — many tracks
        are packed closely together and detection noise is worst.
      • HIGHER alpha (more responsive) when a large genuine jump is detected,
        to avoid lag during sudden accelerations or sharp turns.
      • A jump-detection threshold prevents this from misfiring on jitter.

  Phase 2 (retrospective, every `spline_every` frames):
    Once the per-track EMA history buffer has at least `min_points` entries, a
    natural cubic spline (scipy.interpolate.CubicSpline) is fitted to the entire
    history and the fitted values replace the stored positions. This enforces
    C2-continuity (smooth acceleration curves — physically correct for ground
    vehicles), removes residual EMA noise, and produces analytically exact
    velocity/acceleration estimates by differentiating the spline.

  Crowd density signal:
    Computed each frame as the number of active tracks within `crowd_radius`
    pixels of the current track. This single scalar drives EMA alpha scaling.

MovingAverageSmoother  ← simple fallback
-----------------------------------------
  Simple O(1) online update using a fixed-length ring buffer per track.
  Window size 5–9 frames. Fast but introduces half-window lag and treats
  all frames equally (including jitter frames).

KalmanSmoother  ← alternative for low-jitter scenes
------------------------------------------------------
  1-D constant-velocity Kalman filter applied independently to cx and cy.
  Adapts to measurement noise; smoother on long straight segments but requires
  two noise hyper-parameters to tune.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AdaptiveSplineSmoother  (primary / default)
# ---------------------------------------------------------------------------

class AdaptiveSplineSmoother:
    """
    Hybrid crowd-aware EMA + retrospective cubic-spline trajectory smoother.

    Designed specifically for dense Indian intersection footage where the
    moving-average's fixed lag and equal weighting produce significant velocity
    estimation errors in crowded regions.

    Algorithm overview
    ------------------
    Each frame, for every active track:
      1. Compute local crowd density (tracks within `crowd_radius` pixels).
      2. Derive adaptive EMA alpha from base alpha + density penalty + jump bonus.
      3. Apply EMA update → real-time smoothed (cx, cy) for this frame.
      4. Store smoothed point in the retrospective history buffer.
      5. Every `spline_every` frames (and when a buffer has >= min_points),
         fit a natural cubic spline to the full history and replace stored
         EMA values with spline-evaluated values.  This removes residual
         EMA bias and enforces C2-continuous trajectories.

    Args:
        ema_alpha:       Base EMA responsiveness [0.1 – 0.9].
                         Lower → more smoothing, more lag.
                         Higher → more responsive, less smoothing.
                         Default 0.35 balances lag and smoothness at 25 fps.
        crowd_radius:    Pixel radius used to count nearby tracks for density
                         estimation. Default 150 px (≈3 car lengths at 0.05 m/px).
        crowd_alpha_min: EMA alpha lower-bound applied in maximum crowding.
                         Default 0.15 (heavy smoothing in very dense regions).
        jump_thresh:     Pixel displacement threshold above which a position
                         change is flagged as a genuine rapid motion (not jitter).
                         Alpha is boosted toward 1.0 for such frames.
                         Default 40 px (≈2 m at 0.05 m/px × 1 frame at 25 fps).
        min_points:      Minimum history length before spline fitting is attempted.
                         Cubic spline needs ≥ 4 points; default 8 gives a better
                         initial curve shape.
        spline_every:    Re-fit the spline every N frames.  Lower → smoother
                         trajectory but higher CPU cost.  Default 5 frames.
        history_len:     Maximum history frames per track. Older points are
                         discarded (sliding window).  Default 60 frames (2.4 s
                         at 25 fps).
        max_age:         Frames without update before a track's state is purged.
    """

    def __init__(
        self,
        ema_alpha:       float = 0.35,
        crowd_radius:    float = 150.0,
        crowd_alpha_min: float = 0.15,
        jump_thresh:     float = 40.0,
        min_points:      int   = 8,
        spline_every:    int   = 5,
        history_len:     int   = 60,
        max_age:         int   = 90,
    ) -> None:
        self.ema_alpha       = ema_alpha
        self.crowd_radius    = crowd_radius
        self.crowd_alpha_min = crowd_alpha_min
        self.jump_thresh     = jump_thresh
        self.min_points      = min_points
        self.spline_every    = spline_every
        self.history_len     = history_len
        self.max_age         = max_age

        # Per-track EMA state: last smoothed (cx, cy)
        self._ema_cx: Dict[int, float] = {}
        self._ema_cy: Dict[int, float] = {}

        # Per-track retrospective history: list of (frame_idx, raw_cx, raw_cy)
        # After spline re-fitting these are replaced with spline-evaluated values.
        self._history: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)

        # Per-track spline objects (fitted; None until min_points reached)
        self._spline_x: Dict[int, Optional[CubicSpline]] = {}
        self._spline_y: Dict[int, Optional[CubicSpline]] = {}

        # Frame counter per track (frames since track was last seen)
        self._last_seen:    Dict[int, int] = {}
        self._global_frame: int = 0

        # Frame counter per track for spline re-fitting trigger
        self._frames_since_refit: Dict[int, int] = defaultdict(int)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def update(
        self,
        track_id:      int,
        cx:            float,
        cy:            float,
        all_track_centers: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[float, float]:
        """
        Push a new raw centre observation and return the smoothed estimate.

        Args:
            track_id:           Unique track identifier.
            cx, cy:             Raw centre from the tracker (Kalman output).
            all_track_centers:  List of (cx, cy) for ALL tracks in this frame.
                                Used to compute crowd density.  Pass None to
                                skip density estimation (uses base alpha).

        Returns:
            Smoothed (cx, cy) for this frame.
        """
        self._last_seen[track_id] = self._global_frame

        # ---- Step 1: crowd density → adaptive alpha -------------------------
        alpha = self._compute_alpha(track_id, cx, cy, all_track_centers)

        # ---- Step 2: EMA update --------------------------------------------
        if track_id not in self._ema_cx:
            # First observation: initialise EMA with raw value (zero lag).
            self._ema_cx[track_id] = cx
            self._ema_cy[track_id] = cy
        else:
            self._ema_cx[track_id] = alpha * cx + (1.0 - alpha) * self._ema_cx[track_id]
            self._ema_cy[track_id] = alpha * cy + (1.0 - alpha) * self._ema_cy[track_id]

        s_cx = self._ema_cx[track_id]
        s_cy = self._ema_cy[track_id]

        # ---- Step 3: append to retrospective history -----------------------
        hist = self._history[track_id]
        hist.append((self._global_frame, s_cx, s_cy))

        # Slide window — keep only the most recent `history_len` points.
        if len(hist) > self.history_len:
            self._history[track_id] = hist[-self.history_len:]

        # ---- Step 4: periodic spline re-fitting ----------------------------
        self._frames_since_refit[track_id] += 1
        if (
            len(self._history[track_id]) >= self.min_points
            and self._frames_since_refit[track_id] >= self.spline_every
        ):
            self._refit_spline(track_id)
            self._frames_since_refit[track_id] = 0

        # ---- Step 5: return spline-evaluated position if available ---------
        # The spline is evaluated at the current frame to get the best
        # retrospectively-corrected estimate.
        spx = self._spline_x.get(track_id)
        spy = self._spline_y.get(track_id)
        if spx is not None and spy is not None:
            t_min = self._history[track_id][0][0]
            t_max = self._history[track_id][-1][0]
            t_cur = float(self._global_frame)
            # Clamp to the valid spline domain.
            t_eval = float(np.clip(t_cur, t_min, t_max))
            try:
                return float(spx(t_eval)), float(spy(t_eval))
            except Exception:
                pass  # fall back to EMA if spline evaluation fails

        return s_cx, s_cy

    def tick(self) -> None:
        """
        Advance the internal frame counter and prune stale track states.
        Call once per video frame before processing any tracks in that frame.
        """
        self._global_frame += 1
        stale = [
            tid for tid, last in self._last_seen.items()
            if (self._global_frame - last) > self.max_age
        ]
        for tid in stale:
            self._ema_cx.pop(tid, None)
            self._ema_cy.pop(tid, None)
            self._history.pop(tid, None)
            self._spline_x.pop(tid, None)
            self._spline_y.pop(tid, None)
            self._last_seen.pop(tid, None)
            self._frames_since_refit.pop(tid, None)
        if stale:
            logger.debug("AdaptiveSplineSmoother: pruned %d stale track states.", len(stale))

    def reset(self) -> None:
        """Clear all state (call between video clips for a fresh run)."""
        self._ema_cx.clear()
        self._ema_cy.clear()
        self._history.clear()
        self._spline_x.clear()
        self._spline_y.clear()
        self._last_seen.clear()
        self._frames_since_refit.clear()
        self._global_frame = 0

    def get_velocity(
        self,
        track_id: int,
        fps: float = 25.0,
        scale_m_per_px: float = 0.05,
    ) -> Tuple[float, float, float]:
        """
        Return analytically-derived instantaneous velocity (vx, vy, speed) in m/s
        using the spline derivative at the current frame.

        If no spline has been fitted yet, returns (0.0, 0.0, 0.0).

        Args:
            track_id:        Track whose velocity to compute.
            fps:             Video frame rate for the pixel/frame → m/s conversion.
            scale_m_per_px:  Metres per pixel scale factor.

        Returns:
            (vx, vy, speed) in m/s.
        """
        spx = self._spline_x.get(track_id)
        spy = self._spline_y.get(track_id)
        if spx is None or spy is None:
            return 0.0, 0.0, 0.0

        hist = self._history.get(track_id, [])
        if not hist:
            return 0.0, 0.0, 0.0

        t_min = hist[0][0]
        t_max = hist[-1][0]
        t_cur = float(np.clip(self._global_frame, t_min, t_max))

        try:
            # Spline derivative is in pixels/frame. Convert to m/s.
            vx_px_per_frame = float(spx(t_cur, 1))   # 1st derivative
            vy_px_per_frame = float(spy(t_cur, 1))
            vx = vx_px_per_frame * fps * scale_m_per_px
            vy = vy_px_per_frame * fps * scale_m_per_px
            speed = float(np.hypot(vx, vy))
            return vx, vy, speed
        except Exception:
            return 0.0, 0.0, 0.0

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _compute_alpha(
        self,
        track_id:          int,
        cx:                float,
        cy:                float,
        all_track_centers: Optional[List[Tuple[float, float]]],
    ) -> float:
        """
        Compute the adaptive EMA alpha for this update.

        Logic:
          1. Start from `self.ema_alpha` (base).
          2. Count nearby tracks (crowd density). Scale alpha DOWN linearly
             from base → crowd_alpha_min as density grows from 0 → 10 neighbours.
          3. Detect a genuine large jump (displacement > jump_thresh from the
             current EMA state). Boost alpha toward 1.0 proportionally to the
             excess displacement so the smoother catches up quickly.
        """
        alpha = self.ema_alpha

        # ---- Crowd density penalty -----------------------------------------
        if all_track_centers is not None and len(all_track_centers) > 1:
            n_close = sum(
                1 for (ox, oy) in all_track_centers
                if (ox != cx or oy != cy)  # exclude self
                and np.hypot(ox - cx, oy - cy) <= self.crowd_radius
            )
            # Linear interpolation: 0 neighbours → base alpha; 10+ → min alpha.
            crowd_factor = min(1.0, n_close / 10.0)
            alpha = alpha - crowd_factor * (alpha - self.crowd_alpha_min)

        # ---- Jump boost ----------------------------------------------------
        if track_id in self._ema_cx:
            displacement = np.hypot(
                cx - self._ema_cx[track_id],
                cy - self._ema_cy[track_id],
            )
            if displacement > self.jump_thresh:
                # Boost alpha proportionally: at 2× jump_thresh → alpha = 1.0.
                excess = (displacement - self.jump_thresh) / self.jump_thresh
                jump_alpha = min(1.0, alpha + excess * (1.0 - alpha))
                alpha = jump_alpha
                logger.debug(
                    "AdaptiveSpline: track %d — large jump %.1f px detected, "
                    "alpha boosted to %.3f",
                    track_id, displacement, alpha,
                )

        return float(np.clip(alpha, self.crowd_alpha_min, 1.0))

    def _refit_spline(self, track_id: int) -> None:
        """
        Fit a natural cubic spline to the history of track_id and store it.
        Also replaces historical EMA positions with spline-evaluated values
        to enforce C2-continuity on the stored trajectory.
        """
        hist = self._history[track_id]
        if len(hist) < self.min_points:
            return

        ts  = np.array([h[0] for h in hist], dtype=np.float64)
        cxs = np.array([h[1] for h in hist], dtype=np.float64)
        cys = np.array([h[2] for h in hist], dtype=np.float64)

        # Deduplicate: CubicSpline requires strictly monotonic t values.
        # In rare cases (track re-activation) the same frame index can appear.
        _, unique_idx = np.unique(ts, return_index=True)
        if len(unique_idx) < self.min_points:
            return
        ts  = ts[unique_idx]
        cxs = cxs[unique_idx]
        cys = cys[unique_idx]

        try:
            spx = CubicSpline(ts, cxs, bc_type="natural")
            spy = CubicSpline(ts, cys, bc_type="natural")
        except Exception as e:
            logger.debug("AdaptiveSpline: spline fitting failed for track %d: %s", track_id, e)
            return

        self._spline_x[track_id] = spx
        self._spline_y[track_id] = spy

        # Replace stored EMA positions with spline values (retrospective correction).
        corrected = []
        for (t, _, _) in hist:
            try:
                corrected.append((t, float(spx(float(t))), float(spy(float(t)))))
            except Exception:
                corrected.append((t, float(spx(float(np.clip(t, ts[0], ts[-1])))),
                                     float(spy(float(np.clip(t, ts[0], ts[-1]))))))
        self._history[track_id] = corrected

        logger.debug(
            "AdaptiveSpline: spline fitted for track %d over %d points (t=[%.0f..%.0f])",
            track_id, len(ts), ts[0], ts[-1],
        )


# ---------------------------------------------------------------------------
# Moving-average smoother (kept as fallback)
# ---------------------------------------------------------------------------

class MovingAverageSmoother:
    """
    Per-track simple moving-average (SMA) smoother.

    Maintains a separate position buffer for each track ID.
    Expired tracks (not seen for `max_age` frames) are automatically purged
    to prevent unbounded memory growth.

    Args:
        window:   Number of frames to average (odd values give symmetric lag).
        max_age:  Frames since last update before a track's buffer is dropped.
    """

    def __init__(self, window: int = 7, max_age: int = 60) -> None:
        self.window  = window
        self.max_age = max_age

        self._buffers:      Dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
        self._last_seen:    Dict[int, int]   = {}
        self._global_frame: int = 0

    def update(
        self,
        track_id: int,
        cx:       float,
        cy:       float,
        all_track_centers: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[float, float]:
        """
        Push a new (cx, cy) observation and return the smoothed estimate.

        The `all_track_centers` argument is accepted for API compatibility with
        AdaptiveSplineSmoother but is unused here.
        """
        buf = self._buffers[track_id]
        buf.append((cx, cy))
        self._last_seen[track_id] = self._global_frame

        arr = np.array(buf)
        s_cx = float(np.mean(arr[:, 0]))
        s_cy = float(np.mean(arr[:, 1]))
        return s_cx, s_cy

    def tick(self) -> None:
        """
        Advance the internal frame counter and prune stale buffers.
        Call once per video frame.
        """
        self._global_frame += 1
        stale = [
            tid for tid, last in self._last_seen.items()
            if (self._global_frame - last) > self.max_age
        ]
        for tid in stale:
            del self._buffers[tid]
            del self._last_seen[tid]
        if stale:
            logger.debug("Pruned %d stale smoother buffers.", len(stale))

    def reset(self) -> None:
        """Clear all state (e.g. between video clips)."""
        self._buffers.clear()
        self._last_seen.clear()
        self._global_frame = 0

    def get_velocity(self, track_id: int, fps: float = 25.0,
                     scale_m_per_px: float = 0.05) -> Tuple[float, float, float]:
        """Stub for API compatibility. Returns zeros (velocity not tracked)."""
        return 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# 1-D Kalman smoother (alternative)
# ---------------------------------------------------------------------------

class _Kalman1D:
    """Scalar constant-velocity Kalman filter for a single coordinate."""

    def __init__(self, process_noise: float, measurement_noise: float) -> None:
        self.F  = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.H  = np.array([[1.0, 0.0]])
        self.Q  = np.diag([process_noise, process_noise * 0.1])
        self.R  = np.array([[measurement_noise]])

        self.x  = np.zeros((2, 1))
        self.P  = np.eye(2) * 100.0
        self._initialised = False

    def update(self, z: float) -> float:
        if not self._initialised:
            self.x[0, 0] = z
            self._initialised = True
            return z

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T / S[0, 0]
        self.x = self.x + K * (z - (self.H @ self.x)[0, 0])
        self.P = (np.eye(2) - K @ self.H) @ self.P

        return float(self.x[0, 0])


class KalmanSmoother:
    """
    Per-track Kalman smoother applied independently to cx and cy.

    Args:
        process_noise:      Controls how quickly the filter adapts to genuine
                            motion changes.
        measurement_noise:  Expected pixel-level jitter in raw detections.
        max_age:            Frames without update before purging track state.
    """

    def __init__(
        self,
        process_noise:    float = 0.5,
        measurement_noise: float = 5.0,
        max_age:          int   = 60,
    ) -> None:
        self.process_noise    = process_noise
        self.measurement_noise = measurement_noise
        self.max_age          = max_age

        self._filters_x:  Dict[int, _Kalman1D] = {}
        self._filters_y:  Dict[int, _Kalman1D] = {}
        self._last_seen:  Dict[int, int]        = {}
        self._global_frame = 0

    def _get_or_create(self, track_id: int) -> Tuple[_Kalman1D, _Kalman1D]:
        if track_id not in self._filters_x:
            self._filters_x[track_id] = _Kalman1D(self.process_noise, self.measurement_noise)
            self._filters_y[track_id] = _Kalman1D(self.process_noise, self.measurement_noise)
        return self._filters_x[track_id], self._filters_y[track_id]

    def update(
        self,
        track_id: int,
        cx:       float,
        cy:       float,
        all_track_centers: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[float, float]:
        """Feed a raw position observation and return the Kalman-smoothed estimate."""
        kx, ky = self._get_or_create(track_id)
        self._last_seen[track_id] = self._global_frame
        return kx.update(cx), ky.update(cy)

    def tick(self) -> None:
        """Advance frame counter and prune stale tracks."""
        self._global_frame += 1
        stale = [
            tid for tid, last in self._last_seen.items()
            if (self._global_frame - last) > self.max_age
        ]
        for tid in stale:
            del self._filters_x[tid]
            del self._filters_y[tid]
            del self._last_seen[tid]

    def reset(self) -> None:
        self._filters_x.clear()
        self._filters_y.clear()
        self._last_seen.clear()
        self._global_frame = 0

    def get_velocity(self, track_id: int, fps: float = 25.0,
                     scale_m_per_px: float = 0.05) -> Tuple[float, float, float]:
        """Stub for API compatibility. Returns zeros (velocity not tracked by Kalman)."""
        return 0.0, 0.0, 0.0

"""
Trajectory Smoothing Module

WHY SMOOTHING IS CRITICAL FOR SAFETY ANALYSIS
===============================================
Even after Kalman-filter based tracking, the *output* coordinates still carry
frame-to-frame jitter caused by:
  • Detection bounding-box instability (YOLO's output varies slightly per frame
    even for a stationary object).
  • Quantisation noise from integer pixel rounding.
  • Brief partial occlusions that shift the box by a few pixels.

For safety-rule logic, jitter translates directly into:
  1. Spurious velocity spikes — an object that is actually stationary may appear
     to jump 5 pixels between frames, which at a scale of 0.05 m/px × 25 fps
     gives a phantom velocity of 6.25 m/s (~22 km/h).  This would trigger false
     "speeding" or "sudden acceleration" alarms.
  2. Erratic trajectory curves — the direction-of-travel vector becomes noisy,
     making lane-change or wrong-way detection unreliable.
  3. Incorrect TTC estimates — Time-To-Collision computed from jittery
     positions produces wide confidence intervals.

Smoothing replaces each coordinate with a local temporal average, substantially
reducing high-frequency noise while preserving the genuine low-frequency motion
of vehicles and pedestrians.

THREE IMPLEMENTATIONS
=====================
AdaptiveEMASmoother (recommended — default)
  Exponential Moving Average with adaptive alpha.  The most recent observation
  gets ~60% weight (alpha=0.6), suppressing jitter with near-zero lag.
  When a large position jump is detected (e.g. after track re-activation),
  alpha temporarily spikes to ~0.9 so the estimate snaps to the new observation
  instead of slowly drifting toward it.
  Pro: near-zero lag, preserves sudden manoeuvres, O(1) per update.
  Con: single alpha parameter to tune (0.5–0.8 range).

MovingAverageSmoother (legacy)
  Simple O(1) online update using a fixed-length ring buffer (deque) per track.
  Window size 5–9 frames is sufficient for 25–30 fps video.
  Pro: zero parameters to tune, extremely fast.
  Con: introduces a half-window lag (2–4 frames), acceptable for safety analysis.

KalmanSmoother (alternative)
  1-D Kalman filter applied independently to cx and cy.
  Pro: adapts to measurement noise; smoother on long straight segments.
  Con: requires two noise parameters (process_noise, measurement_noise) to tune.
  Use when you need minimal lag and still want filtering.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adaptive EMA smoother (recommended default)
# ---------------------------------------------------------------------------

class AdaptiveEMASmoother:
    """
    Per-track Exponential Moving Average smoother with adaptive alpha.

    The EMA gives the most recent observation a weight of *alpha* and the
    accumulated history a weight of *(1 − alpha)*.  This provides near-zero
    lag compared to a simple moving average, while still suppressing
    high-frequency jitter from detection instability.

    **Adaptive alpha boost:**  When a new observation is far from the current
    smoothed estimate (e.g. after track re-activation or a large occlusion
    gap), the effective alpha is temporarily increased toward ``alpha_boost``
    so the estimate snaps to the new position instead of slowly averaging
    toward it.  The "jump threshold" is expressed as a multiple of the
    track's typical per-frame displacement, estimated as an EMA of recent
    frame-to-frame deltas.

    Args:
        alpha:           Base smoothing factor (0.0–1.0).  Higher → less
                         smoothing, lower lag.  0.6 is optimal for 25 fps.
        alpha_boost:     Alpha used when a position jump is detected.
        jump_multiplier: A displacement larger than ``jump_multiplier *
                         running_delta`` triggers the alpha boost.
        max_age:         Frames without update before purging track state.
    """

    def __init__(
        self,
        alpha: float = 0.6,
        alpha_boost: float = 0.9,
        jump_multiplier: float = 3.0,
        max_age: int = 120,
    ) -> None:
        self.alpha = alpha
        self.alpha_boost = alpha_boost
        self.jump_multiplier = jump_multiplier
        self.max_age = max_age

        # Per-track smoothed position:  {track_id: (s_cx, s_cy)}
        self._state: Dict[int, Tuple[float, float]] = {}
        # Per-track running EMA of frame-to-frame displacement magnitude
        self._running_delta: Dict[int, float] = {}
        # Bookkeeping for stale-track pruning
        self._last_seen: Dict[int, int] = {}
        self._global_frame: int = 0

    def update(
        self,
        track_id: int,
        cx: float,
        cy: float,
    ) -> Tuple[float, float]:
        """
        Push a new (cx, cy) observation and return the smoothed estimate.

        Args:
            track_id: Unique track identifier.
            cx, cy:   Raw centre coordinates from the tracker.

        Returns:
            Smoothed (cx, cy).
        """
        self._last_seen[track_id] = self._global_frame

        if track_id not in self._state:
            # First observation — initialise without smoothing
            self._state[track_id] = (cx, cy)
            self._running_delta[track_id] = 0.0
            return cx, cy

        prev_cx, prev_cy = self._state[track_id]

        # Frame-to-frame displacement
        displacement = np.hypot(cx - prev_cx, cy - prev_cy)

        # Update running delta estimate (EMA of displacements, α=0.3)
        rd = self._running_delta[track_id]
        rd = 0.3 * displacement + 0.7 * rd
        self._running_delta[track_id] = rd

        # Decide effective alpha: boost if this is a large jump
        # A "large jump" means the displacement exceeds jump_multiplier × the
        # track's typical per-frame movement.  The threshold has a floor of
        # 5.0 px to avoid boosting on normally-moving objects whose running
        # delta happens to be very small.
        threshold = max(5.0, self.jump_multiplier * rd)
        if displacement > threshold:
            effective_alpha = self.alpha_boost
        else:
            effective_alpha = self.alpha

        # EMA update
        s_cx = effective_alpha * cx + (1.0 - effective_alpha) * prev_cx
        s_cy = effective_alpha * cy + (1.0 - effective_alpha) * prev_cy
        self._state[track_id] = (s_cx, s_cy)

        return s_cx, s_cy

    def tick(self) -> None:
        """
        Advance the internal frame counter and prune stale track buffers.
        Call once per video frame.
        """
        self._global_frame += 1
        stale = [
            tid for tid, last in self._last_seen.items()
            if (self._global_frame - last) > self.max_age
        ]
        for tid in stale:
            del self._state[tid]
            del self._running_delta[tid]
            del self._last_seen[tid]
        if stale:
            logger.debug("Pruned %d stale EMA smoother buffers.", len(stale))

    def reset(self) -> None:
        """Clear all state (e.g. between video clips)."""
        self._state.clear()
        self._running_delta.clear()
        self._last_seen.clear()
        self._global_frame = 0


# ---------------------------------------------------------------------------
# Moving-average smoother (legacy)
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

        # {track_id: deque of (cx, cy)}
        self._buffers:       Dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
        # {track_id: frames_since_last_update}
        self._last_seen:     Dict[int, int]   = {}
        self._global_frame:  int = 0

    def update(
        self,
        track_id: int,
        cx:       float,
        cy:       float,
    ) -> Tuple[float, float]:
        """
        Push a new (cx, cy) observation and return the smoothed estimate.

        Args:
            track_id: Unique track identifier.
            cx, cy:   Raw centre coordinates from the tracker.

        Returns:
            Smoothed (cx, cy) — the mean over the buffer.
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
        Call once per video frame (regardless of whether any tracks are active).
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


# ---------------------------------------------------------------------------
# 1-D Kalman smoother (alternative)
# ---------------------------------------------------------------------------

class _Kalman1D:
    """Scalar constant-velocity Kalman filter for a single coordinate."""

    def __init__(self, process_noise: float, measurement_noise: float) -> None:
        # State [x, v], Process model: x' = x + v, v' = v
        self.F  = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.H  = np.array([[1.0, 0.0]])
        self.Q  = np.diag([process_noise, process_noise * 0.1])
        self.R  = np.array([[measurement_noise]])

        self.x  = np.zeros((2, 1))          # state
        self.P  = np.eye(2) * 100.0         # covariance (high initial uncertainty)
        self._initialised = False

    def update(self, z: float) -> float:
        if not self._initialised:
            self.x[0, 0] = z
            self._initialised = True
            return z

        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update
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
                            motion changes. Higher → more responsive but less smooth.
        measurement_noise:  Expected pixel-level jitter in raw detections.
                            Higher → more smoothing but more lag.
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
    ) -> Tuple[float, float]:
        """
        Feed a raw position observation and return the Kalman-smoothed estimate.
        """
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

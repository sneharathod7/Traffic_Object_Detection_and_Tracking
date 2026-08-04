"""
Export Module — Annotated Video + CSV Output

Responsibilities:
  1. Accumulate per-frame track records into an in-memory list.
  2. Compute per-track velocity estimates from consecutive world positions.
  3. Flush everything to a CSV file at end-of-video.
  4. Write a per-frame annotated BGR image to an output video file.
  5. Optionally overlay each track's recent trajectory as a coloured polyline.

CSV Schema
===========
  frame           — 0-based video frame index
  track_id        — unique persistent track identifier
  class_name      — "person" | "car" | "motorcycle" | "bus" | "truck"
  x1, y1, x2, y2 — bounding box corners in pixel coordinates
  center_x        — smoothed pixel column of box centre
  center_y        — smoothed pixel row of box centre
  world_x         — centre_x converted to metres (CoordinateMapper)
  world_y         — centre_y converted to metres
  confidence      — YOLO detection confidence (last matched detection)
  velocity_ms     — estimated speed in m/s (from consecutive world positions × fps)

Video Annotations
==================
  • Colour-coded bounding boxes (class-specific palette; see utils.py).
  • Label: "ID:<n> <class>" rendered above the box with a matching background.
  • Optional trajectory polyline showing the last N smoothed positions.
"""

from __future__ import annotations

import csv
import logging
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils import CLASS_COLORS, draw_box_label

logger = logging.getLogger(__name__)

# CSV header matches the schema documented above
_CSV_HEADER = [
    "frame", "track_id", "class_name",
    "x1", "y1", "x2", "y2",
    "center_x", "center_y",
    "world_x", "world_y",
    "confidence", "velocity_ms",
]


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------

class Exporter:
    """
    Stateful exporter that ingests per-frame track data and writes outputs.

    Usage::

        exporter = Exporter(fps=25, output_video_path="out.mp4",
                            output_csv_path="out.csv", frame_size=(1920, 1080))
        for frame_idx, (frame, tracks) in enumerate(pipeline):
            annotated = exporter.process_frame(frame_idx, frame, tracks)
            # (annotated already written to video internally)
        exporter.close()
    """

    def __init__(
        self,
        fps:               float,
        output_video_path: Optional[str],
        output_csv_path:   str,
        frame_size:        Tuple[int, int],   # (width, height)
        draw_trajectories: bool  = True,
        trajectory_length: int   = 30,        # frames of history to display
        clean_draw:        bool  = False,     # suppress verbose class labels
        append_mode:       bool  = False,     # If True, append to CSV (for resume)
    ) -> None:
        self.fps               = fps
        self.draw_trajectories = draw_trajectories
        self.trajectory_length = trajectory_length
        self.frame_size        = frame_size
        self.clean_draw        = clean_draw

        # ------- CSV writer --------------------------------------------------
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        csv_mode = "a" if append_mode else "w"
        self._csv_file   = open(output_csv_path, csv_mode, newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=_CSV_HEADER)
        if not append_mode:
            self._csv_writer.writeheader()  # Only write header for fresh runs
        logger.info("CSV output (%s mode): %s", csv_mode, output_csv_path)

        # ------- Video writer ------------------------------------------------
        self._video_writer: Optional[cv2.VideoWriter] = None
        if output_video_path:
            Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(
                output_video_path, fourcc, fps, frame_size
            )
            logger.info("Video output: %s", output_video_path)

        # ------- Per-track state for velocity and trajectory -----------------
        # {track_id: (prev_world_x, prev_world_y)}
        self._prev_world: Dict[int, Tuple[float, float]] = {}
        # {track_id: deque[(cx, cy)]}  — smoothed pixel positions for trajectory
        self._traj_buffers: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=trajectory_length)
        )
        # {track_id: deque[float]} — confidence score history
        self._conf_histories: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=5)
        )

        # ------- Failure Clip Recording States -------------------------------
        self._frame_ring: deque[Tuple[int, np.ndarray]] = deque(maxlen=30)
        self._failure_writers: List[Dict] = []
        self._active_motorcycle_tracks: Dict[int, int] = {}  # {track_id: age}

        # Summary counters
        self._total_rows = 0

    # ---- main per-frame API -------------------------------------------------

    def process_frame(
        self,
        frame_idx: int,
        frame:     np.ndarray,
        tracks:    List[Dict],
    ) -> np.ndarray:
        """
        Write CSV rows and produce an annotated copy of *frame*.

        Args:
            frame_idx: 0-based index of the current video frame.
            frame:     Raw BGR image (H × W × 3).
            tracks:    List of track dicts from BYTETracker.update() after
                       smoothing / coordinate mapping is applied.
                       Required keys:
                         track_id, bbox, center, class_name,
                         confidence, world_x, world_y, smoothed_cx, smoothed_cy
                         last_reactivated_frame (optional)

        Returns:
            Annotated BGR image (same size as *frame*).
        """
        # Save raw frame copy to ring buffer for failure clip compiling
        self._frame_ring.append((frame_idx, frame.copy()))
        
        annotated = frame.copy()
        current_motorcycle_ids: Dict[int, int] = {}

        for t in tracks:
            tid        = t["track_id"]
            x1, y1, x2, y2 = t["bbox"]
            s_cx, s_cy = t["smoothed_cx"], t["smoothed_cy"]
            wx, wy     = t["world_x"], t["world_y"]
            conf       = t["confidence"]
            cls_name   = t["class_name"]
            track_age  = t.get("tracklet_len", 1)

            # ---- velocity ---------------------------------------------------
            vel_ms = 0.0
            if tid in self._prev_world:
                pwx, pwy = self._prev_world[tid]
                dist = np.hypot(wx - pwx, wy - pwy)
                vel_ms = float(dist * self.fps)
            self._prev_world[tid] = (wx, wy)

            # ---- CSV row ----------------------------------------------------
            row = {
                "frame":      frame_idx,
                "track_id":   tid,
                "class_name": cls_name,
                "x1":         round(x1, 2),
                "y1":         round(y1, 2),
                "x2":         round(x2, 2),
                "y2":         round(y2, 2),
                "center_x":   round(s_cx, 2),
                "center_y":   round(s_cy, 2),
                "world_x":    round(wx, 4),
                "world_y":    round(wy, 4),
                "confidence": round(conf, 4),
                "velocity_ms": round(vel_ms, 4),
            }
            self._csv_writer.writerow(row)
            self._total_rows += 1

            # ---- trajectory and confidence buffer ---------------------------
            self._traj_buffers[tid].append((int(s_cx), int(s_cy)))
            self._conf_histories[tid].append(conf)

            # ---- track status & color coding --------------------------------
            status = "stable"
            color = CLASS_COLORS.get(cls_name, (200, 200, 200))
            
            if cls_name == "motorcycle":
                current_motorcycle_ids[tid] = track_age
                last_reactivated = t.get("last_reactivated_frame", -1)
                
                # Classify status
                if last_reactivated != -1 and (frame_idx - last_reactivated) < 15:
                    status = "recovered"
                    color = (255, 255, 0)  # Bright Cyan for recovered tracks
                    
                    # Trigger Reconnection Failure debug clip on first detection frame
                    if frame_idx == last_reactivated:
                        self._trigger_failure_clip(frame_idx, tid, "reconnection_recovery")
                elif track_age <= 15:
                    status = "new"
                    color = (255, 0, 255)  # Bright Magenta for newly assigned IDs
                else:
                    status = "stable"
                    color = (0, 165, 255)  # Golden orange for stable motorcycles

            # ---- draw trajectory polyline -----------------------------------
            if self.draw_trajectories:
                pts = list(self._traj_buffers[tid])
                for k in range(1, len(pts)):
                    alpha = k / len(pts)  # fade older points
                    faded = tuple(int(c * alpha) for c in color)
                    cv2.line(annotated, pts[k - 1], pts[k], faded, 1, cv2.LINE_AA)

            # ---- draw bounding box and label --------------------------------
            if self.clean_draw:
                label_text = f"ID:{tid}"
            else:
                label_text = f"ID:{tid} {cls_name}"
                if cls_name == "motorcycle":
                    label_text += f" ({status.upper()})"
                
            draw_box_label(
                annotated,
                bbox      = [int(x1), int(y1), int(x2), int(y2)],
                label     = label_text,
                color     = color,
            )

            # ---- Render Motorcycle Debugging Overlays -----------------------
            if cls_name == "motorcycle" and not self.clean_draw:
                # Draw small status panel below the box
                overlay_y = int(y2) + 12
                conf_list_str = "[" + ",".join([f"{c:.2f}" for c in self._conf_histories[tid]]) + "]"
                debug_info = f"Age:{track_age} | ConfHistory:{conf_list_str}"
                
                cv2.putText(
                    annotated,
                    debug_info,
                    (int(x1), overlay_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                
                # Flashing text warning for recent re-connections
                if status == "recovered" and (frame_idx % 6 < 3):
                    cv2.putText(
                        annotated,
                        "** RECONNECTED ID **",
                        (int(x1), int(y1) - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA
                    )

        # ---- Detect Sudden Motorcycle Loss (Unsupervised Failure) -----------
        for tid, age in self._active_motorcycle_tracks.items():
            if tid not in current_motorcycle_ids:
                # Active motorcycle was lost/disappeared. If it had short duration (fragmentation risk)
                if age < 25:
                    self._trigger_failure_clip(frame_idx, tid, f"track_loss_age_{age}")
                    
        self._active_motorcycle_tracks = current_motorcycle_ids

        # ---- Write Active Failure Video Clips -------------------------------
        self._write_failure_clips(frame_idx, frame)

        # Write annotated frame to video
        if self._video_writer is not None:
            self._video_writer.write(annotated)

        return annotated

    # ---- Failure Clip Recorders ---------------------------------------------

    def _trigger_failure_clip(self, frame_idx: int, track_id: int, event_type: str) -> None:
        """
        Trigger a separate failure recording spanning [frame_idx-30, frame_idx+60] frames.
        """
        output_dir = Path("outputs/debug/motorcycle_failures")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        clip_filename = f"failure_track_{track_id}_frame_{frame_idx}_{event_type}.mp4"
        clip_path = output_dir / clip_filename
        
        # Don't duplicate running clips for same track ID in close temporal vicinity
        for active in self._failure_writers:
            if active["track_id"] == track_id and (frame_idx - active["trigger_frame"]) < 45:
                return
                
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(clip_path),
            fourcc,
            self.fps,
            self.frame_size
        )
        
        # Write frames currently stored in the ring buffer
        start_frame = max(0, frame_idx - 30)
        frames_written = 0
        for f_idx, f_data in self._frame_ring:
            if f_idx >= start_frame:
                writer.write(f_data)
                frames_written += 1
                
        self._failure_writers.append({
            "writer": writer,
            "path": str(clip_path),
            "trigger_frame": frame_idx,
            "end_frame": frame_idx + 60,
            "track_id": track_id,
        })
        logger.info(
            "Triggered failure debug clip for motorcycle track %d on frame %d (pre-cached: %d frames).",
            track_id, frame_idx, frames_written
        )

    def _write_failure_clips(self, frame_idx: int, frame: np.ndarray) -> None:
        """Append the current frame to all active failure video writers."""
        active_writers = []
        for item in self._failure_writers:
            item["writer"].write(frame)
            if frame_idx >= item["end_frame"]:
                item["writer"].release()
                logger.info("Released finished failure debug clip: %s", item["path"])
            else:
                active_writers.append(item)
        self._failure_writers = active_writers

    # ---- cleanup ------------------------------------------------------------

    def close(self) -> None:
        """Flush and close all open output handles."""
        self._csv_file.flush()
        self._csv_file.close()
        logger.info("CSV closed — %d rows written.", self._total_rows)

        if self._video_writer is not None:
            self._video_writer.release()
            logger.info("Video writer released.")
            
        # Clean up any remaining failure writers
        for item in self._failure_writers:
            item["writer"].release()
            logger.info("Closed incomplete failure debug clip: %s", item["path"])
        self._failure_writers.clear()

    def __enter__(self) -> "Exporter":
        return self

    def __exit__(self, *_) -> None:
        self.close()
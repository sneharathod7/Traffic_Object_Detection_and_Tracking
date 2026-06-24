"""
tracking_stability_report.py — Long-Duration Tracking Stability Analysis

Analyzes exported tracking CSV files and generates comprehensive long-duration
performance reports. Designed to answer: "Does tracking quality remain stable
over long time horizons?"

Supports:
  - Single video CSV analysis
  - Multi-video batch analysis (combined report)
  - Time-windowed drift detection
  - Composite health scoring

Usage:
  # Single CSV
  python tracking_stability_report.py --csv outputs/csv/tracks.csv --fps 30

  # Multiple CSVs
  python tracking_stability_report.py --csv-dir outputs/csv/ --fps 30

  # With resurrection log
  python tracking_stability_report.py --csv outputs/csv/tracks.csv --fps 30 \
      --resurrection-log outputs/metrics/resurrection_log.json

  # With switch log
  python tracking_stability_report.py --csv outputs/csv/tracks.csv --fps 30 \
      --switch-log outputs/metrics/switch_log.json

Output:
  outputs/validation/tracking_stability_report.json
  outputs/validation/tracking_stability_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_tracking_csv(csv_path: Path) -> pd.DataFrame:
    """Load and validate a tracking CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"frame", "track_id", "class_name", "x1", "y1", "x2", "y2",
                     "center_x", "center_y", "confidence"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df.sort_values(["frame", "track_id"]).reset_index(drop=True)
    return df


def load_resurrection_log(log_path: Path) -> Dict:
    """Load resurrection log JSON if it exists."""
    if log_path and log_path.exists():
        with open(log_path, "r") as f:
            return json.load(f)
    return {"total_resurrection_attempts": 0, "events": []}


def load_switch_log(log_path: Path) -> Dict:
    """Load switch event log JSON if it exists."""
    if log_path and log_path.exists():
        with open(log_path, "r") as f:
            return json.load(f)
    return {"total_switch_events": 0, "events": []}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Global Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_global_metrics(df: pd.DataFrame, fps: float) -> Dict[str, Any]:
    """Compute aggregate statistics across the entire video."""
    total_frames = int(df["frame"].max() - df["frame"].min() + 1)
    total_tracks = int(df["track_id"].nunique())
    total_detections = len(df)
    duration_seconds = total_frames / fps
    duration_minutes = duration_seconds / 60.0

    # Per-track duration analysis
    track_durations = df.groupby("track_id")["frame"].agg(["min", "max"])
    track_durations["duration_frames"] = track_durations["max"] - track_durations["min"] + 1
    track_durations["duration_seconds"] = track_durations["duration_frames"] / fps

    durations = track_durations["duration_frames"].values

    # Per-class breakdown
    class_tracks = df.groupby("class_name")["track_id"].nunique().to_dict()
    class_detections = df["class_name"].value_counts().to_dict()

    # Per-class average duration
    track_classes = df.groupby("track_id")["class_name"].first()
    track_durations_with_class = track_durations.copy()
    track_durations_with_class["class_name"] = track_classes
    class_avg_duration = track_durations_with_class.groupby("class_name")["duration_frames"].mean().to_dict()

    return {
        "total_frames": total_frames,
        "total_tracks": total_tracks,
        "total_detections": total_detections,
        "video_fps": fps,
        "duration_seconds": round(duration_seconds, 2),
        "duration_minutes": round(duration_minutes, 2),
        "average_track_duration_frames": round(float(np.mean(durations)), 2),
        "median_track_duration_frames": round(float(np.median(durations)), 2),
        "p95_track_duration_frames": round(float(np.percentile(durations, 95)), 2),
        "longest_track_duration_frames": int(np.max(durations)),
        "average_track_duration_seconds": round(float(np.mean(durations)) / fps, 2),
        "median_track_duration_seconds": round(float(np.median(durations)) / fps, 2),
        "tracks_per_class": class_tracks,
        "detections_per_class": class_detections,
        "avg_duration_per_class_frames": {k: round(v, 2) for k, v in class_avg_duration.items()},
        "avg_detections_per_frame": round(total_detections / max(1, total_frames), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Identity Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_identity_metrics(
    df: pd.DataFrame,
    fps: float,
    resurrection_log: Dict,
    switch_log: Dict,
) -> Dict[str, Any]:
    """Compute identity continuity and fragmentation metrics."""
    total_frames = int(df["frame"].max() - df["frame"].min() + 1)
    duration_minutes = total_frames / fps / 60.0

    # Fragmentation: count temporal gaps within each track
    fragmentation_count = 0
    per_class_frag: Dict[str, int] = {}

    for (tid, cls), group in df.groupby(["track_id", "class_name"]):
        frames = sorted(group["frame"].tolist())
        gaps = sum(1 for i in range(1, len(frames)) if frames[i] - frames[i - 1] >= 2)
        if gaps > 0:
            fragmentation_count += gaps
            per_class_frag[cls] = per_class_frag.get(cls, 0) + gaps

    frag_per_minute = fragmentation_count / max(0.01, duration_minutes)

    # Resurrection statistics
    res_events = resurrection_log.get("events", [])
    total_res_attempts = resurrection_log.get("total_resurrection_attempts", len(res_events))
    successful_res = sum(1 for e in res_events if e.get("decision") == "resurrected")
    failed_res = sum(1 for e in res_events if e.get("decision") == "rejected")
    res_success_rate = successful_res / max(1, total_res_attempts) * 100.0

    # Switch events
    switch_events = switch_log.get("events", [])
    total_switches = switch_log.get("total_switch_events", len(switch_events))

    return {
        "fragmentation_count": fragmentation_count,
        "fragmentation_per_minute": round(frag_per_minute, 4),
        "fragmentation_per_class": per_class_frag,
        "resurrection_attempts": total_res_attempts,
        "successful_resurrections": successful_res,
        "failed_resurrections": failed_res,
        "resurrection_success_rate_pct": round(res_success_rate, 2),
        "switch_events_total": total_switches,
        "switch_events_per_minute": round(total_switches / max(0.01, duration_minutes), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Track Health Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_track_health_metrics(df: pd.DataFrame, fps: float) -> Dict[str, Any]:
    """Compute track health, duration distribution, and reliability."""
    total_tracks = df["track_id"].nunique()

    # Duration analysis
    track_durations = df.groupby("track_id")["frame"].agg(["min", "max"])
    track_durations["duration"] = track_durations["max"] - track_durations["min"] + 1
    durations = track_durations["duration"].values

    short_10 = int(np.sum(durations < 10))
    short_30 = int(np.sum(durations < 30))
    medium = int(np.sum((durations >= 30) & (durations < 150)))
    long_tracks = int(np.sum(durations >= 150))

    # Duration distribution (histogram buckets)
    bins = [0, 5, 10, 30, 60, 150, 300, 600, np.inf]
    bin_labels = ["1-5", "6-10", "11-30", "31-60", "61-150", "151-300", "301-600", "600+"]
    hist, _ = np.histogram(durations, bins=bins)
    duration_distribution = dict(zip(bin_labels, [int(x) for x in hist]))

    # Reliability score (from confidence values)
    track_avg_conf = df.groupby("track_id")["confidence"].mean()
    avg_reliability = float(track_avg_conf.mean())

    # Reliability distribution
    rel_bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    rel_labels = ["0.0-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    rel_hist, _ = np.histogram(track_avg_conf.values, bins=rel_bins)
    reliability_distribution = dict(zip(rel_labels, [int(x) for x in rel_hist]))

    return {
        "short_tracks_lt10": short_10,
        "short_tracks_lt10_pct": round(short_10 / max(1, total_tracks) * 100, 2),
        "short_tracks_lt30": short_30,
        "short_tracks_lt30_pct": round(short_30 / max(1, total_tracks) * 100, 2),
        "medium_tracks_30_150": medium,
        "long_tracks_gt150": long_tracks,
        "duration_distribution_frames": duration_distribution,
        "average_reliability_score": round(avg_reliability, 4),
        "reliability_distribution": reliability_distribution,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Time-Windowed Performance Drift
# ═══════════════════════════════════════════════════════════════════════════════

def compute_windowed_metrics(
    df: pd.DataFrame,
    fps: float,
    window_minutes: float = 5.0,
) -> List[Dict[str, Any]]:
    """
    Slice the video into fixed time windows and compute per-window metrics.

    This is the core drift-detection mechanism: if tracking quality degrades
    over time, these per-window metrics will show the deterioration.
    """
    total_frames = int(df["frame"].max() - df["frame"].min() + 1)
    window_frames = int(window_minutes * 60 * fps)
    frame_min = int(df["frame"].min())
    frame_max = int(df["frame"].max())

    windows = []
    window_start = frame_min

    while window_start <= frame_max:
        window_end = min(window_start + window_frames - 1, frame_max)

        # Filter data to this window
        wdf = df[(df["frame"] >= window_start) & (df["frame"] <= window_end)]

        if len(wdf) == 0:
            window_start = window_end + 1
            continue

        w_frames = window_end - window_start + 1
        w_minutes_start = (window_start - frame_min) / fps / 60.0
        w_minutes_end = (window_end - frame_min + 1) / fps / 60.0

        # Tracks active in this window
        w_tracks = wdf["track_id"].nunique()
        w_detections = len(wdf)

        # New tracks born in this window (first appearance is within this window)
        track_first_frame = df.groupby("track_id")["frame"].min()
        new_tracks = int(((track_first_frame >= window_start) & (track_first_frame <= window_end)).sum())

        # Tracks that end in this window (last appearance is within this window)
        track_last_frame = df.groupby("track_id")["frame"].max()
        ended_tracks = int(((track_last_frame >= window_start) & (track_last_frame <= window_end)).sum())

        # Fragmentation in this window
        w_frag = 0
        for tid, group in wdf.groupby("track_id"):
            frames = sorted(group["frame"].tolist())
            w_frag += sum(1 for i in range(1, len(frames)) if frames[i] - frames[i - 1] >= 2)

        # Short tracks starting in this window
        tracks_born_here = track_first_frame[
            (track_first_frame >= window_start) & (track_first_frame <= window_end)
        ].index
        if len(tracks_born_here) > 0:
            born_durations = df[df["track_id"].isin(tracks_born_here)].groupby("track_id")["frame"].agg(
                ["min", "max"]
            )
            born_durations["dur"] = born_durations["max"] - born_durations["min"] + 1
            short_born = int((born_durations["dur"] < 10).sum())
        else:
            short_born = 0

        # Average track duration for tracks active in this window
        active_durations = wdf.groupby("track_id")["frame"].agg(["min", "max"])
        active_durations["dur"] = active_durations["max"] - active_durations["min"] + 1
        avg_track_len = float(active_durations["dur"].mean())

        # Average confidence in this window
        avg_conf = float(wdf["confidence"].mean())

        windows.append({
            "window_label": f"min_{w_minutes_start:.1f}_to_{w_minutes_end:.1f}",
            "frame_start": int(window_start),
            "frame_end": int(window_end),
            "minutes_start": round(w_minutes_start, 2),
            "minutes_end": round(w_minutes_end, 2),
            "total_frames": w_frames,
            "active_tracks": w_tracks,
            "new_tracks_born": new_tracks,
            "tracks_ended": ended_tracks,
            "detections": w_detections,
            "fragmentation_gaps": w_frag,
            "fragmentation_rate_per_100f": round(w_frag / max(1, w_tracks) / max(1, w_frames) * 100, 4),
            "short_tracks_born_lt10": short_born,
            "avg_track_length_in_window": round(avg_track_len, 2),
            "avg_confidence": round(avg_conf, 4),
        })

        window_start = window_end + 1

    return windows


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Composite Scores
# ═══════════════════════════════════════════════════════════════════════════════

def compute_tracking_health_score(
    global_metrics: Dict,
    identity_metrics: Dict,
    health_metrics: Dict,
    windowed_metrics: List[Dict],
) -> Dict[str, Any]:
    """
    Compute a single 0–100 Tracking Health Score.

    Formula:
      THS = w1 * frag_score + w2 * duration_score + w3 * resurrection_score
          + w4 * short_track_score + w5 * stability_score

    Components (each 0–100):
      1. frag_score (w=0.25):
         100 - min(100, frag_per_minute * 10)
         Penalizes fragmentation rate.

      2. duration_score (w=0.25):
         min(100, avg_track_duration_seconds / target_seconds * 100)
         Rewards longer average track durations. Target = 10s.

      3. resurrection_score (w=0.15):
         If no attempts → 100 (nothing to recover).
         Else: resurrection_success_rate.

      4. short_track_score (w=0.15):
         100 - min(100, short_tracks_lt10_pct * 2.5)
         Penalizes high percentage of short tracks.

      5. stability_score (w=0.20):
         Based on coefficient of variation of per-window fragmentation rates.
         Lower variation → higher stability.
         100 - min(100, CV * 100)
    """
    weights = {
        "fragmentation": 0.25,
        "duration": 0.25,
        "resurrection": 0.15,
        "short_tracks": 0.15,
        "stability": 0.20,
    }

    # 1. Fragmentation score
    frag_per_min = identity_metrics.get("fragmentation_per_minute", 0)
    frag_score = max(0, 100 - min(100, frag_per_min * 10))

    # 2. Duration score (target: 10 seconds average track duration)
    avg_dur_s = global_metrics.get("average_track_duration_seconds", 0)
    target_dur_s = 10.0
    duration_score = min(100, avg_dur_s / target_dur_s * 100)

    # 3. Resurrection score
    res_attempts = identity_metrics.get("resurrection_attempts", 0)
    if res_attempts == 0:
        resurrection_score = 100.0  # Nothing to recover — perfect
    else:
        resurrection_score = identity_metrics.get("resurrection_success_rate_pct", 0)

    # 4. Short track score
    short_pct = health_metrics.get("short_tracks_lt10_pct", 0)
    short_track_score = max(0, 100 - min(100, short_pct * 2.5))

    # 5. Stability score (CV of per-window fragmentation rates)
    if len(windowed_metrics) >= 2:
        frag_rates = [w["fragmentation_rate_per_100f"] for w in windowed_metrics]
        frag_mean = np.mean(frag_rates)
        frag_std = np.std(frag_rates)
        cv = frag_std / max(0.001, frag_mean)
        stability_score = max(0, 100 - min(100, cv * 100))
    else:
        stability_score = 100.0  # Single window → stable by definition

    # Composite
    ths = (
        weights["fragmentation"] * frag_score
        + weights["duration"] * duration_score
        + weights["resurrection"] * resurrection_score
        + weights["short_tracks"] * short_track_score
        + weights["stability"] * stability_score
    )

    return {
        "tracking_health_score": round(ths, 2),
        "components": {
            "fragmentation_score": round(frag_score, 2),
            "duration_score": round(duration_score, 2),
            "resurrection_score": round(resurrection_score, 2),
            "short_track_score": round(short_track_score, 2),
            "stability_score": round(stability_score, 2),
        },
        "weights": weights,
        "formula": (
            "THS = 0.25*frag + 0.25*duration + 0.15*resurrection"
            " + 0.15*short_tracks + 0.20*stability"
        ),
    }


def compute_identity_continuity_score(
    identity_metrics: Dict,
    health_metrics: Dict,
    global_metrics: Dict,
) -> Dict[str, Any]:
    """
    Compute a 0–100 Identity Continuity Score focused exclusively
    on how well the tracker preserves identity over time.

    Formula:
      ICS = w1 * frag_component + w2 * resurrection_component
          + w3 * short_track_component + w4 * identity_stability_component

    Components (each 0–100):
      1. frag_component (w=0.35):
         100 - min(100, frag_per_minute * 8)
         Penalizes identity breaks.

      2. resurrection_component (w=0.20):
         If no attempts → 100.
         Else: success_rate.

      3. short_track_component (w=0.25):
         100 - min(100, short_tracks_lt10_pct * 3)
         Short tracks are strong indicators of identity failure.

      4. identity_stability_component (w=0.20):
         Based on switch_events_per_minute.
         100 - min(100, switches_per_min * 15)
    """
    weights = {
        "fragmentation": 0.35,
        "resurrection": 0.20,
        "short_tracks": 0.25,
        "identity_stability": 0.20,
    }

    # 1. Fragmentation
    frag_per_min = identity_metrics.get("fragmentation_per_minute", 0)
    frag_component = max(0, 100 - min(100, frag_per_min * 8))

    # 2. Resurrection
    res_attempts = identity_metrics.get("resurrection_attempts", 0)
    if res_attempts == 0:
        resurrection_component = 100.0
    else:
        resurrection_component = identity_metrics.get("resurrection_success_rate_pct", 0)

    # 3. Short tracks
    short_pct = health_metrics.get("short_tracks_lt10_pct", 0)
    short_track_component = max(0, 100 - min(100, short_pct * 3))

    # 4. Identity stability (switch events per minute)
    switches_per_min = identity_metrics.get("switch_events_per_minute", 0)
    identity_stability = max(0, 100 - min(100, switches_per_min * 15))

    ics = (
        weights["fragmentation"] * frag_component
        + weights["resurrection"] * resurrection_component
        + weights["short_tracks"] * short_track_component
        + weights["identity_stability"] * identity_stability
    )

    return {
        "identity_continuity_score": round(ics, 2),
        "components": {
            "fragmentation_component": round(frag_component, 2),
            "resurrection_component": round(resurrection_component, 2),
            "short_track_component": round(short_track_component, 2),
            "identity_stability_component": round(identity_stability, 2),
        },
        "weights": weights,
        "formula": (
            "ICS = 0.35*frag + 0.20*resurrection"
            " + 0.25*short_tracks + 0.20*identity_stability"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Warnings Engine
# ═══════════════════════════════════════════════════════════════════════════════

def generate_warnings(
    global_metrics: Dict,
    identity_metrics: Dict,
    health_metrics: Dict,
    windowed_metrics: List[Dict],
    ths: Dict,
    ics: Dict,
) -> List[Dict[str, str]]:
    """Generate human-readable warnings for concerning metrics."""
    warnings = []

    # High fragmentation rate
    if identity_metrics["fragmentation_per_minute"] > 5.0:
        warnings.append({
            "severity": "HIGH",
            "metric": "fragmentation_per_minute",
            "value": identity_metrics["fragmentation_per_minute"],
            "message": f"Fragmentation rate ({identity_metrics['fragmentation_per_minute']:.2f}/min) "
                       f"exceeds threshold of 5.0/min. Identity breaks are frequent.",
        })

    # Too many short tracks
    if health_metrics["short_tracks_lt10_pct"] > 25.0:
        warnings.append({
            "severity": "HIGH",
            "metric": "short_tracks_lt10_pct",
            "value": health_metrics["short_tracks_lt10_pct"],
            "message": f"{health_metrics['short_tracks_lt10_pct']:.1f}% of tracks are shorter than "
                       f"10 frames. This suggests detection instability or aggressive ID creation.",
        })

    # Low health score
    if ths["tracking_health_score"] < 60:
        warnings.append({
            "severity": "HIGH",
            "metric": "tracking_health_score",
            "value": ths["tracking_health_score"],
            "message": f"Tracking Health Score ({ths['tracking_health_score']:.1f}/100) is below "
                       f"the acceptable threshold of 60. System may not be production-ready.",
        })

    # Low identity continuity
    if ics["identity_continuity_score"] < 60:
        warnings.append({
            "severity": "HIGH",
            "metric": "identity_continuity_score",
            "value": ics["identity_continuity_score"],
            "message": f"Identity Continuity Score ({ics['identity_continuity_score']:.1f}/100) is "
                       f"below 60. Identity preservation is insufficient.",
        })

    # Stability drift detection
    if len(windowed_metrics) >= 3:
        frag_rates = [w["fragmentation_rate_per_100f"] for w in windowed_metrics]
        # Check if last window is significantly worse than first
        if len(frag_rates) >= 2 and frag_rates[-1] > frag_rates[0] * 2.0 and frag_rates[-1] > 0.01:
            warnings.append({
                "severity": "MEDIUM",
                "metric": "fragmentation_drift",
                "value": f"first={frag_rates[0]:.4f}, last={frag_rates[-1]:.4f}",
                "message": "Fragmentation rate doubled between first and last window. "
                           "Possible tracker degradation over time.",
            })

    # High switch rate
    if identity_metrics["switch_events_per_minute"] > 3.0:
        warnings.append({
            "severity": "MEDIUM",
            "metric": "switch_events_per_minute",
            "value": identity_metrics["switch_events_per_minute"],
            "message": f"Switch event rate ({identity_metrics['switch_events_per_minute']:.2f}/min) "
                       f"is elevated. May indicate association instability.",
        })

    return warnings


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Report Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_json_report(
    video_name: str,
    global_metrics: Dict,
    identity_metrics: Dict,
    health_metrics: Dict,
    windowed_metrics: List[Dict],
    ths: Dict,
    ics: Dict,
    warnings: List[Dict],
) -> Dict:
    """Assemble the complete JSON report."""
    return {
        "report_version": "1.0.0",
        "generated_at": datetime.now().isoformat(),
        "video_name": video_name,
        "duration_minutes": global_metrics["duration_minutes"],
        "global_metrics": global_metrics,
        "identity_metrics": identity_metrics,
        "track_health_metrics": health_metrics,
        "tracking_health_score": ths,
        "identity_continuity_score": ics,
        "window_metrics": windowed_metrics,
        "warnings": warnings,
    }


def generate_markdown_report(report: Dict) -> str:
    """Generate a professional human-readable markdown report."""
    gm = report["global_metrics"]
    im = report["identity_metrics"]
    hm = report["track_health_metrics"]
    ths = report["tracking_health_score"]
    ics = report["identity_continuity_score"]
    wins = report["window_metrics"]
    warns = report["warnings"]

    lines = []
    lines.append("# 📊 Tracking Stability Report")
    lines.append("")
    lines.append(f"**Video:** `{report['video_name']}`")
    lines.append(f"**Generated:** {report['generated_at']}")
    lines.append(f"**Duration:** {gm['duration_minutes']:.1f} minutes ({gm['total_frames']} frames @ {gm['video_fps']} FPS)")
    lines.append("")

    # ── Executive Summary ──
    lines.append("---")
    lines.append("## Executive Summary")
    lines.append("")

    ths_val = ths["tracking_health_score"]
    ics_val = ics["identity_continuity_score"]
    grade = "🟢 Excellent" if ths_val >= 80 else "🟡 Acceptable" if ths_val >= 60 else "🔴 Needs Improvement"

    lines.append(f"| Metric | Score | Grade |")
    lines.append(f"|--------|-------|-------|")
    lines.append(f"| **Tracking Health Score** | **{ths_val:.1f}** / 100 | {grade} |")

    ics_grade = "🟢 Strong" if ics_val >= 80 else "🟡 Moderate" if ics_val >= 60 else "🔴 Weak"
    lines.append(f"| **Identity Continuity Score** | **{ics_val:.1f}** / 100 | {ics_grade} |")
    lines.append("")

    if warns:
        lines.append(f"> [!WARNING]")
        lines.append(f"> {len(warns)} warning(s) detected. See Warnings section below.")
        lines.append("")
    else:
        lines.append(f"> [!TIP]")
        lines.append(f"> No warnings detected. Tracking quality is within acceptable bounds.")
        lines.append("")

    # ── Global Metrics ──
    lines.append("---")
    lines.append("## Global Metrics")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total Frames | {gm['total_frames']:,} |")
    lines.append(f"| Total Tracks | {gm['total_tracks']:,} |")
    lines.append(f"| Total Detections | {gm['total_detections']:,} |")
    lines.append(f"| Avg Detections/Frame | {gm['avg_detections_per_frame']:.1f} |")
    lines.append(f"| Avg Track Duration | {gm['average_track_duration_frames']:.1f} frames ({gm['average_track_duration_seconds']:.2f}s) |")
    lines.append(f"| Median Track Duration | {gm['median_track_duration_frames']:.1f} frames ({gm['median_track_duration_seconds']:.2f}s) |")
    lines.append(f"| 95th Percentile Duration | {gm['p95_track_duration_frames']:.1f} frames |")
    lines.append(f"| Longest Track | {gm['longest_track_duration_frames']:,} frames |")
    lines.append("")

    # Per-class breakdown
    lines.append("### Tracks Per Class")
    lines.append("")
    lines.append("| Class | Tracks | Detections | Avg Duration (frames) |")
    lines.append("|-------|--------|------------|----------------------|")
    for cls in sorted(gm.get("tracks_per_class", {}).keys()):
        t = gm["tracks_per_class"].get(cls, 0)
        d = gm["detections_per_class"].get(cls, 0)
        ad = gm["avg_duration_per_class_frames"].get(cls, 0)
        lines.append(f"| {cls} | {t} | {d:,} | {ad:.1f} |")
    lines.append("")

    # ── Identity Metrics ──
    lines.append("---")
    lines.append("## Identity Metrics")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Fragmentation Count | {im['fragmentation_count']} |")
    lines.append(f"| Fragmentation / Minute | {im['fragmentation_per_minute']:.4f} |")
    lines.append(f"| Switch Events | {im['switch_events_total']} |")
    lines.append(f"| Switches / Minute | {im['switch_events_per_minute']:.4f} |")
    lines.append("")

    # Fragmentation per class
    if im.get("fragmentation_per_class"):
        lines.append("### Fragmentation Per Class")
        lines.append("")
        lines.append("| Class | Gaps |")
        lines.append("|-------|------|")
        for cls, gaps in sorted(im["fragmentation_per_class"].items()):
            lines.append(f"| {cls} | {gaps} |")
        lines.append("")

    # ── Resurrection Statistics ──
    lines.append("---")
    lines.append("## Resurrection Statistics")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total Attempts | {im['resurrection_attempts']} |")
    lines.append(f"| Successful | {im['successful_resurrections']} |")
    lines.append(f"| Rejected | {im['failed_resurrections']} |")
    lines.append(f"| Success Rate | {im['resurrection_success_rate_pct']:.1f}% |")
    lines.append("")

    # ── Track Health ──
    lines.append("---")
    lines.append("## Track Health")
    lines.append("")
    lines.append(f"| Metric | Count | Percentage |")
    lines.append(f"|--------|-------|------------|")
    lines.append(f"| Short Tracks (<10 frames) | {hm['short_tracks_lt10']} | {hm['short_tracks_lt10_pct']:.1f}% |")
    lines.append(f"| Short Tracks (<30 frames) | {hm['short_tracks_lt30']} | {hm['short_tracks_lt30_pct']:.1f}% |")
    lines.append(f"| Medium Tracks (30-150) | {hm['medium_tracks_30_150']} | — |")
    lines.append(f"| Long Tracks (>150) | {hm['long_tracks_gt150']} | — |")
    lines.append(f"| Avg Reliability | {hm['average_reliability_score']:.4f} | — |")
    lines.append("")

    # Duration distribution
    lines.append("### Duration Distribution (frames)")
    lines.append("")
    lines.append("| Bucket | Count |")
    lines.append("|--------|-------|")
    for bucket, count in hm["duration_distribution_frames"].items():
        lines.append(f"| {bucket} | {count} |")
    lines.append("")

    # ── Tracking Health Score ──
    lines.append("---")
    lines.append("## Tracking Health Score")
    lines.append("")
    lines.append(f"**Score: {ths['tracking_health_score']:.1f} / 100**")
    lines.append("")
    lines.append(f"```")
    lines.append(f"Formula: {ths['formula']}")
    lines.append(f"```")
    lines.append("")
    lines.append(f"| Component | Score | Weight |")
    lines.append(f"|-----------|-------|--------|")
    for comp_name, comp_val in ths["components"].items():
        w = ths["weights"].get(comp_name.replace("_score", "").replace("_component", ""), "—")
        lines.append(f"| {comp_name} | {comp_val:.1f} | {w} |")
    lines.append("")

    # ── Identity Continuity Score ──
    lines.append("---")
    lines.append("## Identity Continuity Score")
    lines.append("")
    lines.append(f"**Score: {ics['identity_continuity_score']:.1f} / 100**")
    lines.append("")
    lines.append(f"```")
    lines.append(f"Formula: {ics['formula']}")
    lines.append(f"```")
    lines.append("")
    lines.append(f"| Component | Score | Weight |")
    lines.append(f"|-----------|-------|--------|")
    for comp_name, comp_val in ics["components"].items():
        w = ics["weights"].get(comp_name.replace("_component", ""), "—")
        lines.append(f"| {comp_name} | {comp_val:.1f} | {w} |")
    lines.append("")

    # ── Performance Drift (Windowed) ──
    lines.append("---")
    lines.append("## Performance Drift Analysis (Time Windows)")
    lines.append("")

    if len(wins) >= 2:
        lines.append("| Window | Active Tracks | New Tracks | Frag Gaps | Frag Rate | Short Born | Avg Track Len | Avg Conf |")
        lines.append("|--------|---------------|------------|-----------|-----------|------------|---------------|----------|")
        for w in wins:
            lines.append(
                f"| {w['window_label']} | {w['active_tracks']} | {w['new_tracks_born']} "
                f"| {w['fragmentation_gaps']} | {w['fragmentation_rate_per_100f']:.4f} "
                f"| {w['short_tracks_born_lt10']} | {w['avg_track_length_in_window']:.1f} "
                f"| {w['avg_confidence']:.4f} |"
            )
        lines.append("")
    else:
        lines.append("*Video too short for multi-window analysis.*")
        lines.append("")

    # ── Warnings ──
    lines.append("---")
    lines.append("## Warnings")
    lines.append("")
    if warns:
        for w in warns:
            severity_icon = "🔴" if w["severity"] == "HIGH" else "🟡"
            lines.append(f"- {severity_icon} **[{w['severity']}]** `{w['metric']}`: {w['message']}")
        lines.append("")
    else:
        lines.append("> [!TIP]")
        lines.append("> No warnings. All metrics are within acceptable bounds.")
        lines.append("")

    # ── Recommendations ──
    lines.append("---")
    lines.append("## Recommendations")
    lines.append("")
    if ths["tracking_health_score"] >= 80 and ics["identity_continuity_score"] >= 80:
        lines.append("- ✅ System is ready for production deployment on long-duration videos.")
        lines.append("- ✅ No parameter changes recommended.")
    elif ths["tracking_health_score"] >= 60:
        lines.append("- 🟡 System is acceptable but could benefit from further optimization.")
        if hm["short_tracks_lt10_pct"] > 20:
            lines.append("- Consider investigating the high short-track rate — this may indicate detection instability in certain regions.")
        if im["fragmentation_per_minute"] > 3:
            lines.append("- Fragmentation rate is elevated. Consider reviewing the track buffer configuration for this scene density.")
    else:
        lines.append("- 🔴 System needs investigation before production deployment.")
        lines.append("- Review the per-window drift analysis to identify when quality degrades.")
        lines.append("- Consider running the merge audit on the output CSV.")
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: Multi-Video Combined Report
# ═══════════════════════════════════════════════════════════════════════════════

def generate_combined_report(individual_reports: List[Dict]) -> Dict:
    """Aggregate metrics across multiple videos into a combined report."""
    combined = {
        "report_version": "1.0.0",
        "generated_at": datetime.now().isoformat(),
        "video_count": len(individual_reports),
        "videos": [],
        "aggregate_metrics": {},
    }

    # Collect per-video summaries
    total_frames = 0
    total_tracks = 0
    total_detections = 0
    total_frag = 0
    total_duration_min = 0.0
    all_ths = []
    all_ics = []
    all_warnings = []

    for r in individual_reports:
        gm = r["global_metrics"]
        im = r["identity_metrics"]
        ths = r["tracking_health_score"]
        ics = r["identity_continuity_score"]

        total_frames += gm["total_frames"]
        total_tracks += gm["total_tracks"]
        total_detections += gm["total_detections"]
        total_frag += im["fragmentation_count"]
        total_duration_min += gm["duration_minutes"]
        all_ths.append(ths["tracking_health_score"])
        all_ics.append(ics["identity_continuity_score"])
        all_warnings.extend(r.get("warnings", []))

        combined["videos"].append({
            "video_name": r["video_name"],
            "duration_minutes": gm["duration_minutes"],
            "total_tracks": gm["total_tracks"],
            "tracking_health_score": ths["tracking_health_score"],
            "identity_continuity_score": ics["identity_continuity_score"],
            "fragmentation_count": im["fragmentation_count"],
            "warning_count": len(r.get("warnings", [])),
        })

    combined["aggregate_metrics"] = {
        "total_frames_processed": total_frames,
        "total_tracks_created": total_tracks,
        "total_detections_ingested": total_detections,
        "total_duration_minutes": round(total_duration_min, 2),
        "total_fragmentation_events": total_frag,
        "mean_tracking_health_score": round(float(np.mean(all_ths)), 2),
        "min_tracking_health_score": round(float(np.min(all_ths)), 2),
        "max_tracking_health_score": round(float(np.max(all_ths)), 2),
        "mean_identity_continuity_score": round(float(np.mean(all_ics)), 2),
        "min_identity_continuity_score": round(float(np.min(all_ics)), 2),
        "max_identity_continuity_score": round(float(np.max(all_ics)), 2),
        "total_warnings": len(all_warnings),
    }

    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_single_video(
    csv_path: Path,
    fps: float,
    resurrection_log_path: Optional[Path] = None,
    switch_log_path: Optional[Path] = None,
    window_minutes: float = 5.0,
) -> Dict:
    """Run the full stability analysis on a single video's tracking CSV."""
    logger.info("Loading tracking data from %s", csv_path)
    df = load_tracking_csv(csv_path)

    resurrection_log = load_resurrection_log(resurrection_log_path) if resurrection_log_path else {
        "total_resurrection_attempts": 0, "events": []
    }
    switch_log = load_switch_log(switch_log_path) if switch_log_path else {
        "total_switch_events": 0, "events": []
    }

    logger.info("Computing global metrics...")
    global_metrics = compute_global_metrics(df, fps)

    logger.info("Computing identity metrics...")
    identity_metrics = compute_identity_metrics(df, fps, resurrection_log, switch_log)

    logger.info("Computing track health metrics...")
    health_metrics = compute_track_health_metrics(df, fps)

    logger.info("Computing windowed performance drift metrics (window=%.1f min)...", window_minutes)
    windowed_metrics = compute_windowed_metrics(df, fps, window_minutes)

    logger.info("Computing Tracking Health Score...")
    ths = compute_tracking_health_score(global_metrics, identity_metrics, health_metrics, windowed_metrics)

    logger.info("Computing Identity Continuity Score...")
    ics = compute_identity_continuity_score(identity_metrics, health_metrics, global_metrics)

    logger.info("Generating warnings...")
    warnings = generate_warnings(global_metrics, identity_metrics, health_metrics, windowed_metrics, ths, ics)

    video_name = csv_path.stem
    report = generate_json_report(
        video_name, global_metrics, identity_metrics, health_metrics,
        windowed_metrics, ths, ics, warnings,
    )

    logger.info("Tracking Health Score: %.1f / 100", ths["tracking_health_score"])
    logger.info("Identity Continuity Score: %.1f / 100", ics["identity_continuity_score"])
    if warnings:
        for w in warnings:
            logger.warning("[%s] %s", w["severity"], w["message"])
    else:
        logger.info("No warnings detected.")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tracking_stability_report",
        description="Long-Duration Tracking Stability Analysis Suite",
    )
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to a single tracking CSV file.")
    parser.add_argument("--csv-dir", type=str, default=None,
                        help="Path to a directory of tracking CSV files (batch mode).")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Video FPS (used for time conversions).")
    parser.add_argument("--resurrection-log", type=str, default=None,
                        help="Path to resurrection_log.json.")
    parser.add_argument("--switch-log", type=str, default=None,
                        help="Path to switch_log.json.")
    parser.add_argument("--window-minutes", type=float, default=5.0,
                        help="Window size in minutes for drift analysis.")
    parser.add_argument("--output-dir", type=str, default="outputs/validation",
                        help="Output directory for reports.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    res_log = Path(args.resurrection_log) if args.resurrection_log else None
    sw_log = Path(args.switch_log) if args.switch_log else None

    if args.csv:
        # Single video mode
        report = analyze_single_video(
            Path(args.csv), args.fps, res_log, sw_log, args.window_minutes,
        )

        # Save JSON
        json_path = output_dir / "tracking_stability_report.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("JSON report saved to %s", json_path)

        # Save Markdown
        md_content = generate_markdown_report(report)
        md_path = output_dir / "tracking_stability_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info("Markdown report saved to %s", md_path)

    elif args.csv_dir:
        # Batch mode
        csv_dir = Path(args.csv_dir)
        csv_files = sorted(csv_dir.glob("*.csv"))
        if not csv_files:
            logger.error("No CSV files found in %s", csv_dir)
            sys.exit(1)

        individual_reports = []
        for csv_file in csv_files:
            logger.info("=" * 60)
            logger.info("Processing: %s", csv_file.name)
            report = analyze_single_video(
                csv_file, args.fps, res_log, sw_log, args.window_minutes,
            )
            individual_reports.append(report)

            # Save individual report
            ind_json = output_dir / f"{csv_file.stem}_stability.json"
            with open(ind_json, "w") as f:
                json.dump(report, f, indent=2)

            ind_md = output_dir / f"{csv_file.stem}_stability.md"
            md_content = generate_markdown_report(report)
            with open(ind_md, "w", encoding="utf-8") as f:
                f.write(md_content)

        # Combined report
        combined = generate_combined_report(individual_reports)
        combined_path = output_dir / "combined_tracking_report.json"
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2)
        logger.info("Combined report saved to %s", combined_path)

    else:
        parser.error("Must specify either --csv or --csv-dir")


if __name__ == "__main__":
    main()

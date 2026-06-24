"""
tracking_drift_analysis.py — Long-Duration Tracking Drift Detection

Detects whether the tracker deteriorates during long videos by fitting
simple linear trend lines to per-window metrics and classifying each
metric trend as: Improving, Stable, or Degrading.

This is a standalone analysis tool that consumes either:
  - A tracking_stability_report.json (from tracking_stability_report.py)
  - Or directly, a tracking CSV + FPS

Measurements:
  1. Fragmentation Trend
  2. Resurrection Trend (if resurrection_log available)
  3. Short Track Trend
  4. Average Track Length Trend
  5. Reliability (Confidence) Trend

Usage:
  # From a stability report JSON
  python tracking_drift_analysis.py --report outputs/validation/tracking_stability_report.json

  # Directly from CSV
  python tracking_drift_analysis.py --csv outputs/csv/tracks.csv --fps 30

Output:
  outputs/validation/tracking_drift_analysis.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Trend Classification
# ═══════════════════════════════════════════════════════════════════════════════

def fit_trend(
    values: List[float],
    metric_name: str,
    higher_is_better: bool = False,
    stability_threshold: float = 0.01,
) -> Dict[str, Any]:
    """
    Fit a linear trend to a sequence of per-window metric values.

    Parameters:
      values: List of metric values, one per time window (chronological order).
      metric_name: Human-readable name of the metric.
      higher_is_better: If True, an increasing slope is "Improving".
                        If False, an increasing slope is "Degrading".
      stability_threshold: Absolute slope below this is classified as "Stable".

    Returns:
      Dict with slope, classification, R², and raw values.
    """
    n = len(values)
    if n < 2:
        return {
            "metric": metric_name,
            "classification": "INSUFFICIENT_DATA",
            "reason": f"Only {n} window(s) available. Need >= 2 for trend analysis.",
            "values": values,
            "slope": None,
            "r_squared": None,
        }

    x = np.arange(n, dtype=float)
    y = np.array(values, dtype=float)

    # Handle all-zero or constant series
    if np.std(y) < 1e-10:
        return {
            "metric": metric_name,
            "classification": "STABLE",
            "reason": "Metric is constant across all windows.",
            "values": values,
            "slope": 0.0,
            "r_squared": 1.0,
            "intercept": float(y[0]),
            "first_value": float(y[0]),
            "last_value": float(y[-1]),
            "change_pct": 0.0,
        }

    # Linear regression: y = slope * x + intercept
    coeffs = np.polyfit(x, y, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    # R² calculation
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = float(1.0 - ss_res / max(ss_tot, 1e-10))

    # Normalize slope relative to mean to get a scale-invariant measure
    mean_val = float(np.mean(y))
    if abs(mean_val) > 1e-10:
        normalized_slope = slope / abs(mean_val)
    else:
        normalized_slope = slope

    # Classification
    if abs(normalized_slope) < stability_threshold:
        classification = "STABLE"
        reason = (
            f"Normalized slope ({normalized_slope:.6f}) is within "
            f"stability threshold (±{stability_threshold})."
        )
    elif normalized_slope > 0:
        if higher_is_better:
            classification = "IMPROVING"
            reason = f"Metric is increasing (slope={slope:.6f}, R²={r_squared:.4f})."
        else:
            classification = "DEGRADING"
            reason = f"Metric is increasing (slope={slope:.6f}, R²={r_squared:.4f}). Higher values indicate degradation."
    else:
        if higher_is_better:
            classification = "DEGRADING"
            reason = f"Metric is decreasing (slope={slope:.6f}, R²={r_squared:.4f}). Lower values indicate degradation."
        else:
            classification = "IMPROVING"
            reason = f"Metric is decreasing (slope={slope:.6f}, R²={r_squared:.4f})."

    # Percentage change from first to last window
    first_val = float(y[0])
    last_val = float(y[-1])
    if abs(first_val) > 1e-10:
        change_pct = (last_val - first_val) / abs(first_val) * 100.0
    else:
        change_pct = 0.0 if abs(last_val) < 1e-10 else float("inf")

    return {
        "metric": metric_name,
        "classification": classification,
        "reason": reason,
        "values": [round(v, 6) for v in values],
        "slope": round(slope, 8),
        "normalized_slope": round(normalized_slope, 8),
        "intercept": round(intercept, 6),
        "r_squared": round(r_squared, 6),
        "first_value": round(first_val, 6),
        "last_value": round(last_val, 6),
        "change_pct": round(change_pct, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Drift Analysis from Windowed Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def run_drift_analysis(window_metrics: List[Dict]) -> Dict[str, Any]:
    """
    Run full drift analysis on a list of per-window metric dictionaries.

    Each window dict should have keys:
      - fragmentation_gaps
      - fragmentation_rate_per_100f
      - short_tracks_born_lt10
      - avg_track_length_in_window
      - avg_confidence
      - new_tracks_born
      - active_tracks
    """
    if len(window_metrics) < 2:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": f"Only {len(window_metrics)} window(s). Need >= 2 for drift analysis.",
            "trends": [],
        }

    trends = []

    # 1. Fragmentation Trend (lower is better)
    frag_values = [w.get("fragmentation_rate_per_100f", 0) for w in window_metrics]
    trends.append(fit_trend(frag_values, "Fragmentation Rate", higher_is_better=False))

    # 2. Short Track Trend (lower is better)
    short_values = [w.get("short_tracks_born_lt10", 0) for w in window_metrics]
    trends.append(fit_trend(short_values, "Short Tracks Born (<10f)", higher_is_better=False))

    # 3. Average Track Length Trend (higher is better)
    avg_len_values = [w.get("avg_track_length_in_window", 0) for w in window_metrics]
    trends.append(fit_trend(avg_len_values, "Average Track Length", higher_is_better=True))

    # 4. Reliability / Confidence Trend (higher is better)
    conf_values = [w.get("avg_confidence", 0) for w in window_metrics]
    trends.append(fit_trend(conf_values, "Average Confidence", higher_is_better=True))

    # 5. New Tracks Born Trend (neutral — but sudden spikes indicate instability)
    new_track_values = [w.get("new_tracks_born", 0) for w in window_metrics]
    trends.append(fit_trend(new_track_values, "New Tracks Born", higher_is_better=False,
                            stability_threshold=0.05))

    # 6. Active Tracks Trend (informational)
    active_values = [w.get("active_tracks", 0) for w in window_metrics]
    trends.append(fit_trend(active_values, "Active Tracks", higher_is_better=True,
                            stability_threshold=0.05))

    # Overall classification
    degrading = [t for t in trends if t["classification"] == "DEGRADING"]
    improving = [t for t in trends if t["classification"] == "IMPROVING"]
    stable = [t for t in trends if t["classification"] == "STABLE"]

    if len(degrading) >= 2:
        overall = "DEGRADING"
        overall_msg = (
            f"{len(degrading)} metric(s) show degradation over time: "
            f"{', '.join(t['metric'] for t in degrading)}. "
            f"Tracker stability may be compromised for long-duration videos."
        )
    elif len(degrading) == 1 and len(improving) == 0:
        overall = "SLIGHTLY_DEGRADING"
        overall_msg = (
            f"1 metric shows mild degradation: {degrading[0]['metric']}. "
            f"Monitor closely on longer videos."
        )
    elif len(improving) > len(degrading):
        overall = "IMPROVING"
        overall_msg = (
            f"{len(improving)} metric(s) are improving over time. "
            f"Tracker warms up and stabilizes."
        )
    else:
        overall = "STABLE"
        overall_msg = (
            f"{len(stable)} metric(s) are stable. "
            f"Tracking quality is consistent across the video duration."
        )

    return {
        "status": overall,
        "message": overall_msg,
        "window_count": len(window_metrics),
        "trends": trends,
        "summary": {
            "degrading_count": len(degrading),
            "improving_count": len(improving),
            "stable_count": len(stable),
            "degrading_metrics": [t["metric"] for t in degrading],
            "improving_metrics": [t["metric"] for t in improving],
            "stable_metrics": [t["metric"] for t in stable],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Markdown Report
# ═══════════════════════════════════════════════════════════════════════════════

def generate_drift_markdown(drift_result: Dict, video_name: str = "") -> str:
    """Generate a human-readable markdown drift analysis report."""
    lines = []
    lines.append("# 📈 Tracking Drift Analysis Report")
    lines.append("")
    if video_name:
        lines.append(f"**Video:** `{video_name}`")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    lines.append(f"**Windows Analyzed:** {drift_result.get('window_count', 0)}")
    lines.append("")

    # Overall verdict
    status = drift_result["status"]
    icon = {"STABLE": "🟢", "IMPROVING": "🟢", "SLIGHTLY_DEGRADING": "🟡", "DEGRADING": "🔴",
            "INSUFFICIENT_DATA": "⚪"}.get(status, "⚪")

    lines.append("---")
    lines.append("## Overall Verdict")
    lines.append("")
    lines.append(f"**{icon} {status}**")
    lines.append("")
    lines.append(f"> {drift_result['message']}")
    lines.append("")

    # Trends table
    trends = drift_result.get("trends", [])
    if trends:
        lines.append("---")
        lines.append("## Metric Trends")
        lines.append("")
        lines.append("| Metric | Classification | Slope | R² | First → Last | Change % |")
        lines.append("|--------|---------------|-------|-----|-------------|----------|")

        for t in trends:
            cls = t["classification"]
            cls_icon = {"STABLE": "🟢", "IMPROVING": "🟢", "DEGRADING": "🔴",
                        "SLIGHTLY_DEGRADING": "🟡", "INSUFFICIENT_DATA": "⚪"}.get(cls, "⚪")

            slope = f"{t['slope']:.6f}" if t["slope"] is not None else "—"
            r2 = f"{t['r_squared']:.4f}" if t["r_squared"] is not None else "—"
            first_last = f"{t.get('first_value', '—')} → {t.get('last_value', '—')}"
            change = f"{t.get('change_pct', 0):.1f}%" if t.get("change_pct") is not None else "—"

            lines.append(
                f"| {t['metric']} | {cls_icon} {cls} | {slope} | {r2} | {first_last} | {change} |"
            )
        lines.append("")

        # Detailed breakdown
        lines.append("---")
        lines.append("## Detailed Trend Analysis")
        lines.append("")
        for t in trends:
            cls_icon = {"STABLE": "🟢", "IMPROVING": "🟢", "DEGRADING": "🔴",
                        "SLIGHTLY_DEGRADING": "🟡", "INSUFFICIENT_DATA": "⚪"}.get(
                            t["classification"], "⚪")
            lines.append(f"### {cls_icon} {t['metric']}")
            lines.append("")
            lines.append(f"- **Classification:** {t['classification']}")
            lines.append(f"- **Reason:** {t['reason']}")
            if t.get("values"):
                vals_str = ", ".join(f"{v:.4f}" for v in t["values"])
                lines.append(f"- **Per-Window Values:** `[{vals_str}]`")
            lines.append("")

    # Summary
    summary = drift_result.get("summary", {})
    if summary:
        lines.append("---")
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Degrading metrics:** {summary.get('degrading_count', 0)}")
        lines.append(f"- **Improving metrics:** {summary.get('improving_count', 0)}")
        lines.append(f"- **Stable metrics:** {summary.get('stable_count', 0)}")
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tracking_drift_analysis",
        description="Detect whether tracking quality degrades over time.",
    )
    parser.add_argument("--report", type=str, default=None,
                        help="Path to a tracking_stability_report.json (preferred input).")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to a tracking CSV (alternative input — requires --fps).")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Video FPS (only used with --csv).")
    parser.add_argument("--window-minutes", type=float, default=5.0,
                        help="Window size in minutes (only used with --csv).")
    parser.add_argument("--output-dir", type=str, default="outputs/validation",
                        help="Output directory.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_name = ""

    if args.report:
        # Load from stability report
        report_path = Path(args.report)
        if not report_path.exists():
            logger.error("Report not found: %s", report_path)
            sys.exit(1)

        with open(report_path, "r") as f:
            report = json.load(f)

        window_metrics = report.get("window_metrics", [])
        video_name = report.get("video_name", report_path.stem)

    elif args.csv:
        # Compute windowed metrics directly from CSV
        import pandas as pd
        from tracking_stability_report import load_tracking_csv, compute_windowed_metrics

        csv_path = Path(args.csv)
        df = load_tracking_csv(csv_path)
        window_metrics = compute_windowed_metrics(df, args.fps, args.window_minutes)
        video_name = csv_path.stem

    else:
        parser.error("Must specify either --report or --csv")
        return

    logger.info("Running drift analysis on %d windows...", len(window_metrics))
    drift_result = run_drift_analysis(window_metrics)

    # Save JSON
    json_path = output_dir / "tracking_drift_analysis.json"
    with open(json_path, "w") as f:
        json.dump(drift_result, f, indent=2)
    logger.info("Drift analysis JSON saved to %s", json_path)

    # Save Markdown
    md_content = generate_drift_markdown(drift_result, video_name)
    md_path = output_dir / "tracking_drift_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Drift analysis markdown saved to %s", md_path)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  DRIFT ANALYSIS: {drift_result['status']}")
    print(f"{'=' * 60}")
    print(f"  {drift_result['message']}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

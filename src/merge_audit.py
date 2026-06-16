import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from reid import ReIDExtractor
from tracker import compute_appearance_distance

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

def compute_kinematics_and_scale(group: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Compute velocity anomaly, acceleration anomaly, and scale anomaly.
    Returns normalized anomaly scores [0, 1].
    """
    frames = group["frame"].values
    cx = group["center_x"].values
    cy = group["center_y"].values
    w = group["x2"].values - group["x1"].values
    h = group["y2"].values - group["y1"].values
    areas = w * h

    if len(frames) < 3:
        return 0.0, 0.0, 0.0

    # Velocity (px/frame)
    dx = np.diff(cx)
    dy = np.diff(cy)
    dt = np.diff(frames)
    
    vx = dx / dt
    vy = dy / dt
    speeds = np.hypot(vx, vy)
    
    vel_anomaly = 0.0
    if len(speeds) > 2:
        rolling_std = pd.Series(speeds).rolling(window=5, min_periods=3).std().fillna(0).values
        # Flag: velocity_jump > 3 * rolling_velocity_std AND jump > 30.0 px/frame (teleport)
        speed_diffs = np.abs(np.diff(speeds))
        thresholds = np.maximum(3 * rolling_std[:-1], 30.0)
        jumps = speed_diffs - thresholds
        if np.any(jumps > 0):
            vel_anomaly = min(1.0, np.max(jumps) / 30.0)

    # Acceleration
    ax = np.diff(vx) / dt[1:]
    ay = np.diff(vy) / dt[1:]
    accel = np.hypot(ax, ay)
    # physical_threshold for acceleration: e.g. 20 px/frame^2 (huge acceleration)
    accel_anomaly = 0.0
    if len(accel) > 0 and np.any(accel > 20.0):
        accel_anomaly = min(1.0, (np.max(accel) - 20.0) / 20.0)

    # Scale anomaly
    area_ratio = areas[1:] / areas[:-1]
    scale_jumps = np.maximum(area_ratio, 1.0 / area_ratio)
    scale_anomaly = 0.0
    # Area jump > 2.0 (doubling or halving instantly)
    if np.any(scale_jumps > 2.0):
        scale_anomaly = min(1.0, np.max(scale_jumps) - 2.0)

    return max(vel_anomaly, accel_anomaly), 0.0, scale_anomaly

def compute_heading_anomaly(group: pd.DataFrame) -> float:
    """
    Compute max delta heading between consecutive segments.
    """
    frames = group["frame"].values
    cx = group["center_x"].values
    cy = group["center_y"].values

    if len(frames) < 10:
        return 0.0

    # Smooth positions to compute stable headings
    smooth_cx = pd.Series(cx).rolling(window=5, min_periods=1).mean().values
    smooth_cy = pd.Series(cy).rolling(window=5, min_periods=1).mean().values
    
    dx = np.diff(smooth_cx)
    dy = np.diff(smooth_cy)
    
    headings = np.arctan2(dy, dx)
    
    # Delta heading
    delta_h = np.diff(headings)
    # Wrap to [-pi, pi]
    delta_h = (delta_h + np.pi) % (2 * np.pi) - np.pi
    
    max_delta_deg = np.rad2deg(np.max(np.abs(delta_h)))
    
    # Anomaly > 120 deg is highly anomalous
    if max_delta_deg > 120.0:
        return min(1.0, (max_delta_deg - 120.0) / 60.0)
    return 0.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="outputs/metrics/suspicious_tracks.json")
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    total_tracks = df["track_id"].nunique()
    logger.info("Loaded %d tracks from %s", total_tracks, args.csv)

    # We will sample every N frames for appearance to save time
    SAMPLE_INTERVAL = 5
    sampled_df = df[df["frame"] % SAMPLE_INTERVAL == 0].copy()
    
    frames_to_read = sorted(sampled_df["frame"].unique())
    logger.info("Extracting appearance features for %d frames...", len(frames_to_read))

    reid = ReIDExtractor(device="cuda" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(args.video)
    
    # Dictionary to store embeddings per track: track_id -> List of dicts (frame, emb)
    track_embeddings = {tid: [] for tid in df["track_id"].unique()}

    frame_idx = 0
    pbar = tqdm(total=len(frames_to_read), desc="Extracting ReID")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx in frames_to_read:
            frame_dets = sampled_df[sampled_df["frame"] == frame_idx]
            if len(frame_dets) > 0:
                crops = []
                tids = []
                for _, row in frame_dets.iterrows():
                    x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    if x2 - x1 > 5 and y2 - y1 > 5:
                        crop = frame[y1:y2, x1:x2]
                        crops.append(crop)
                        tids.append(row["track_id"])
                
                if crops:
                    embs = reid.extract_combined(crops)
                    for tid, emb in zip(tids, embs):
                        track_embeddings[tid].append({"frame": frame_idx, "emb": emb})
            pbar.update(1)
            
        frame_idx += 1

    cap.release()
    pbar.close()

    logger.info("Computing consistency scores...")
    
    audit_results = []
    
    for (tid, cls_name), group in df.groupby(["track_id", "class_name"]):
        group = group.sort_values("frame")
        
        vel_anomaly, _, scale_anomaly = compute_kinematics_and_scale(group)
        hdg_anomaly = compute_heading_anomaly(group)
        
        # Appearance anomaly
        embs = track_embeddings[tid]
        embs = sorted(embs, key=lambda x: x["frame"])
        app_anomaly = 0.0
        if len(embs) >= 2:
            distances = []
            for i in range(1, len(embs)):
                d = compute_appearance_distance(embs[i]["emb"], embs[i-1]["emb"])
                distances.append(d)
            # Threshold for anomaly: mean distance > 0.40 is suspicious
            mean_d = np.mean(distances)
            if mean_d > 0.40:
                app_anomaly = min(1.0, (mean_d - 0.40) / 0.40)

        suspicion_score = (
            0.4 * app_anomaly +
            0.3 * hdg_anomaly +
            0.2 * vel_anomaly +
            0.1 * scale_anomaly
        )
        
        reasons = []
        if app_anomaly > 0.0: reasons.append("appearance_jump")
        if hdg_anomaly > 0.0: reasons.append("heading_jump")
        if vel_anomaly > 0.0: reasons.append("velocity_jump")
        if scale_anomaly > 0.0: reasons.append("scale_jump")

        if suspicion_score > 0.0:
            audit_results.append({
                "track_id": int(tid),
                "class_name": cls_name,
                "duration": len(group),
                "suspicion_score": round(float(suspicion_score), 4),
                "appearance_anomaly": round(float(app_anomaly), 4),
                "heading_anomaly": round(float(hdg_anomaly), 4),
                "velocity_anomaly": round(float(vel_anomaly), 4),
                "scale_anomaly": round(float(scale_anomaly), 4),
                "reasons": reasons
            })

    # Sort by suspicion score
    audit_results.sort(key=lambda x: x["suspicion_score"], reverse=True)
    
    # Filter only those that actually have some anomaly
    suspicious_tracks = [x for x in audit_results if x["suspicion_score"] > 0.2]
    
    report = {
        "total_tracks": total_tracks,
        "suspicious_tracks_count": len(suspicious_tracks),
        "suspicious_percentage": round(len(suspicious_tracks) / total_tracks * 100, 2),
        "top_20_suspicious_tracks": suspicious_tracks[:20]
    }
    
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
        
    logger.info("Merge Audit Complete!")
    logger.info("Total Tracks: %d", total_tracks)
    logger.info("Suspicious Tracks: %d (%.2f%%)", len(suspicious_tracks), report["suspicious_percentage"])
    
    if suspicious_tracks:
        reasons_flat = [r for t in suspicious_tracks for r in t["reasons"]]
        most_common = max(set(reasons_flat), key=reasons_flat.count) if reasons_flat else "None"
        logger.info("Most common anomaly: %s", most_common)
        
        logger.info("\nTop 10 Suspicious Tracks:")
        for t in suspicious_tracks[:10]:
            logger.info("  Track %3d (%-10s) | Score: %.3f | Reasons: %s", 
                        t["track_id"], t["class_name"], t["suspicion_score"], ",".join(t["reasons"]))
    else:
        logger.info("No highly suspicious tracks found!")

if __name__ == "__main__":
    main()

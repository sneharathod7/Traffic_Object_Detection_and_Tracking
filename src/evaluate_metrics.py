import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import cv2
import sys
import os
import motmetrics as mm
from pathlib import Path
from collections import deque

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Put your file paths here (relative to project root or custom paths):
XML_PATH     = str(PROJECT_ROOT / "ground_truth" / "groundtruth.xml")
RAW_PRED_CSV = str(PROJECT_ROOT / "outputs" / "csv" / "full1_tracks.csv")
OUT_CSV      = str(PROJECT_ROOT / "ground_truth" / "ground_truth.csv")
OUT_VIDEO    = str(PROJECT_ROOT / "ground_truth" / "ground_truth_preview_cleandraw.mp4")
VIDEO_INPUT  = str(PROJECT_ROOT / "data" / "video" / "full1.MP4")
METRICS_TXT  = str(PROJECT_ROOT / "ground_truth" / "metrics_report.txt")

MAX_FRAME    = 299
IOU_THRESH   = 0.5  # For matching prediction bounding boxes to ground truth

CLASS_COLORS = {
    "person": (0, 0, 255),
    "car": (255, 0, 0),
    "motorcycle": (0, 255, 0),
    "bus": (0, 255, 255),
    "truck": (255, 0, 255),
}

# ─── PARSE CVAT XML ───────────────────────────────────────────────────────────
print("1. Parsing CVAT XML...")
if not os.path.exists(XML_PATH):
    print(f"Error: Could not find {XML_PATH}")
    sys.exit(1)

tree = ET.parse(XML_PATH)
root = tree.getroot()

rows = []
for track in root.findall('track'):
    track_id = int(track.get('id'))
    if track_id == 48:
        continue  # Exclude ID 48 as requested
    class_name = track.get('label')
    
    # Process all boxes. In CVAT XML, all interpolated frames are explicitly written.
    for box in track.findall('box'):
        if box.get('outside') == '1':
            continue  # The object left the frame
        frame_idx = int(box.get('frame'))
        if frame_idx > MAX_FRAME:
            continue
            
        xtl = float(box.get('xtl'))
        ytl = float(box.get('ytl'))
        xbr = float(box.get('xbr'))
        ybr = float(box.get('ybr'))
        
        rows.append({
            "frame": frame_idx,
            "track_id": track_id,
            "class_name": class_name,
            "x1": xtl,
            "y1": ytl,
            "x2": xbr,
            "y2": ybr,
            "center_x": (xtl + xbr) / 2,
            "center_y": (ytl + ybr) / 2,
            "confidence": 1.0
        })

df_gt = pd.DataFrame(rows)
df_gt = df_gt.sort_values(by=["frame", "track_id"]).reset_index(drop=True)

# Save to CSV
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df_gt.to_csv(OUT_CSV, index=False)
print(f"   -> Saved ground truth to {OUT_CSV}")

# ─── CALCULATE METRICS ────────────────────────────────────────────────────────
print("\n2. Calculating Tracking & Detection Metrics...")
df_pred = pd.read_csv(RAW_PRED_CSV)
df_pred = df_pred[df_pred["frame"] <= MAX_FRAME].copy()

# Ensure coordinates are numeric
for col in ["x1", "y1", "x2", "y2"]:
    df_gt[col] = pd.to_numeric(df_gt[col])
    df_pred[col] = pd.to_numeric(df_pred[col])

# Create MOT accumulators for overall and per-class
accs = {}
classes = df_gt["class_name"].unique().tolist()
# Also add "OVERALL"
classes_to_eval = ["OVERALL"] + classes

for cls in classes_to_eval:
    accs[cls] = mm.MOTAccumulator(auto_id=True)

# Helper for IoU Distance
def box_iou(b1, b2):
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (b1[2]-b1[0]) * (b1[3]-b1[1])
    boxBArea = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = boxAArea + boxBArea - inter
    return inter / float(union + 1e-7)

for frame_idx in range(MAX_FRAME + 1):
    gt_frame = df_gt[df_gt["frame"] == frame_idx]
    pred_frame = df_pred[df_pred["frame"] == frame_idx]
    
    for cls in classes_to_eval:
        if cls == "OVERALL":
            gts = gt_frame
            preds = pred_frame
        else:
            gts = gt_frame[gt_frame["class_name"] == cls]
            preds = pred_frame[pred_frame["class_name"] == cls]
            
        gt_ids = gts["track_id"].tolist()
        pred_ids = preds["track_id"].tolist()
        
        # Build distance matrix (1 - IoU). If IoU < threshold, distance is NaN
        dists = []
        for _, g in gts.iterrows():
            row_dists = []
            boxG = [g["x1"], g["y1"], g["x2"], g["y2"]]
            for _, p in preds.iterrows():
                boxP = [p["x1"], p["y1"], p["x2"], p["y2"]]
                iou = box_iou(boxG, boxP)
                if iou >= IOU_THRESH:
                    row_dists.append(1 - iou)
                else:
                    row_dists.append(np.nan)
            dists.append(row_dists)
            
        accs[cls].update(
            gt_ids,
            pred_ids,
            dists
        )

# ─── ADDITIONAL TRACKING METRICS ─────────────────────────────────────────────
# Track Fragmentation: For each GT track, count how many separate "runs" of
# matched predictions exist. If a track goes matched → unmatched → matched again,
# that is 1 fragmentation event.
print("Computing advanced tracking metrics (Fragmentation, Track Quality)...")

def compute_track_fragmentation(df_gt: pd.DataFrame, df_pred: pd.DataFrame, iou_thresh: float = 0.5):
    """For each GT track, count how many times predictions break and resume."""
    frag_data = {}
    for gt_id, gt_group in df_gt.groupby("track_id"):
        cls = gt_group.iloc[0]["class_name"]
        gt_frames = sorted(gt_group["frame"].unique())
        total_gt_frames = len(gt_frames)

        matched_frames = set()
        for f in gt_frames:
            gt_row = gt_group[gt_group["frame"] == f].iloc[0]
            boxG = [gt_row["x1"], gt_row["y1"], gt_row["x2"], gt_row["y2"]]
            pred_at_f = df_pred[(df_pred["frame"] == f) & (df_pred["class_name"] == cls)]
            for _, p in pred_at_f.iterrows():
                boxP = [p["x1"], p["y1"], p["x2"], p["y2"]]
                if box_iou(boxG, boxP) >= iou_thresh:
                    matched_frames.add(f)
                    break

        # Count fragmentation: number of transitions from matched→unmatched within track
        frags = 0
        was_matched = None
        for f in gt_frames:
            is_matched = f in matched_frames
            if was_matched is True and not is_matched:
                frags += 1
            was_matched = is_matched

        # Track quality
        pct = len(matched_frames) / total_gt_frames if total_gt_frames > 0 else 0.0
        if pct >= 0.80:
            quality = "MT"   # Mostly Tracked
        elif pct >= 0.20:
            quality = "PT"   # Partially Tracked
        else:
            quality = "ML"   # Mostly Lost

        frag_data[gt_id] = {
            "class_name": cls,
            "total_gt_frames": total_gt_frames,
            "matched_frames": len(matched_frames),
            "coverage_pct": round(pct * 100, 1),
            "fragmentations": frags,
            "quality": quality,
        }
    return frag_data

frag_data = compute_track_fragmentation(df_gt, df_pred, IOU_THRESH)

# Summarise fragmentation stats
total_frags = sum(v["fragmentations"] for v in frag_data.values())
mt_count = sum(1 for v in frag_data.values() if v["quality"] == "MT")
pt_count = sum(1 for v in frag_data.values() if v["quality"] == "PT")
ml_count = sum(1 for v in frag_data.values() if v["quality"] == "ML")
frag_per_class = {}
for v in frag_data.values():
    cls = v["class_name"]
    frag_per_class.setdefault(cls, {"MT": 0, "PT": 0, "ML": 0, "frags": 0})
    frag_per_class[cls][v["quality"]] += 1
    frag_per_class[cls]["frags"] += v["fragmentations"]

# Compute metrics
mh = mm.metrics.create()
report_lines = []
json_output = {}

report_lines.append("="*80)
report_lines.append("              GROUND TRUTH vs. PREDICTIONS (Raw Output) REPORT")
report_lines.append("="*80)

# 1. Total Unique Vehicles Count
report_lines.append("\n[ UNIQUE VEHICLES COUNTED ]")
overall_gt_unique = df_gt["track_id"].nunique()
overall_pred_unique = df_pred["track_id"].nunique()
report_lines.append(f"OVERALL : GT = {overall_gt_unique} vehicles | PRED = {overall_pred_unique} vehicles")
json_output["unique_vehicles"] = {"overall_gt": overall_gt_unique, "overall_pred": overall_pred_unique, "by_class": {}}

for cls in classes:
    gt_cls_unique = df_gt[df_gt["class_name"] == cls]["track_id"].nunique()
    pred_cls_unique = df_pred[df_pred["class_name"] == cls]["track_id"].nunique()
    report_lines.append(f"  - {cls.capitalize()} : GT = {gt_cls_unique} | PRED = {pred_cls_unique}")
    json_output["unique_vehicles"]["by_class"][cls] = {"gt": gt_cls_unique, "pred": pred_cls_unique}

# 2. Track Quality Distribution
report_lines.append("\n[ TRACK QUALITY DISTRIBUTION ]")
report_lines.append(f"  Mostly Tracked  (MT, >=80% coverage) : {mt_count}  tracks")
report_lines.append(f"  Partially Tracked (PT, 20-80%)      : {pt_count}  tracks")
report_lines.append(f"  Mostly Lost     (ML, <20% coverage) : {ml_count}  tracks")
report_lines.append(f"  Total Track Fragmentations          : {total_frags}")
report_lines.append("")
report_lines.append("  Per-Class Breakdown:")
report_lines.append(f"  {'Class':<12} {'MT':>5} {'PT':>5} {'ML':>5} {'Frags':>7}")
report_lines.append("  " + "-"*38)
for cls, d in sorted(frag_per_class.items()):
    report_lines.append(f"  {cls.capitalize():<12} {d['MT']:>5} {d['PT']:>5} {d['ML']:>5} {d['frags']:>7}")

json_output["track_quality"] = {
    "mostly_tracked": mt_count, "partially_tracked": pt_count, "mostly_lost": ml_count,
    "total_fragmentations": total_frags, "by_class": frag_per_class
}

# 3. MOT + Detection metrics per class
report_lines.append("\n[ TRACKING & DETECTION METRICS ]")
summary_cols = ['mota', 'motp', 'idf1', 'num_switches', 'num_false_positives', 'num_misses', 'num_objects', 'mostly_tracked', 'mostly_lost']
json_output["metrics"] = {}

for cls in classes_to_eval:
    acc = accs[cls]
    if len(acc.mot_events) == 0 or len(acc.events.index) == 0:
        continue

    summary = mh.compute(acc, metrics=summary_cols, name=cls)

    num_gt_boxes = summary.loc[cls, 'num_objects']
    FN = summary.loc[cls, 'num_misses']
    FP = summary.loc[cls, 'num_false_positives']
    TP = num_gt_boxes - FN

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / num_gt_boxes if num_gt_boxes > 0 else 0.0
    f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mota      = summary.loc[cls, 'mota'] * 100
    motp      = summary.loc[cls, 'motp']                         # mean 1-IoU of matched pairs (lower = better overlap)
    idf1      = summary.loc[cls, 'idf1'] * 100
    id_switches = summary.loc[cls, 'num_switches']
    cls_frags = frag_per_class.get(cls, {}).get("frags", "N/A") if cls != "OVERALL" else total_frags

    report_lines.append(f"\n--- {cls.upper()} ---")
    report_lines.append(f"  F1 Score                     : {f1:.4f}")
    report_lines.append(f"  Precision                    : {precision:.4f}")
    report_lines.append(f"  Recall                       : {recall:.4f}")
    report_lines.append(f"  MOTA (Tracking Accuracy)     : {mota:.2f}%")
    report_lines.append(f"  MOTP (Bounding Box Quality)  : {(1-motp)*100:.2f}% avg IoU on matched boxes")
    report_lines.append(f"  IDF1 (ID Consistency)        : {idf1:.2f}%")
    report_lines.append(f"  ID Switches                  : {id_switches}")
    report_lines.append(f"  Track Fragmentations         : {cls_frags}")
    report_lines.append(f"  False Positives (FP)         : {FP}")
    report_lines.append(f"  False Negatives (FN)         : {FN}")
    report_lines.append(f"  True Positives  (TP)         : {TP}")
    report_lines.append(f"  Total GT Boxes               : {num_gt_boxes}")

    json_output["metrics"][cls] = {
        "f1": round(f1, 4), "precision": round(precision, 4), "recall": round(recall, 4),
        "mota_pct": round(float(mota), 2), "motp_avg_iou_pct": round(float((1-motp)*100), 2),
        "idf1_pct": round(float(idf1), 2), "id_switches": int(id_switches),
        "fragmentations": int(cls_frags) if isinstance(cls_frags, (int, float)) else cls_frags,
        "FP": int(FP), "FN": int(FN), "TP": int(TP), "total_gt_boxes": int(num_gt_boxes)
    }

report_lines.append("\n" + "="*80)

# Print, save TXT, save JSON
report_text = "\n".join(report_lines)
print(report_text)
with open(METRICS_TXT, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"   -> Saved metrics report (.txt) to {METRICS_TXT}")

import json
json_path = METRICS_TXT.replace(".txt", ".json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(json_output, f, indent=2)
print(f"   -> Saved metrics report (.json) to {json_path}")

print("Exiting before slow video rendering...")
sys.exit(0)

# ─── RENDER GROUND TRUTH VIDEO ────────────────────────────────────────────────
print("\n3. Rendering Ground Truth Video...")

cap = cv2.VideoCapture(VIDEO_INPUT)
if not cap.isOpened():
    print(f"Error: Cannot open video {VIDEO_INPUT}")
    sys.exit(1)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (width, height))

def draw_box_label(img, bbox, label, color):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(img, (x1, y1 - text_h - baseline - 5), (x1 + text_w, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - baseline - 2), font, font_scale, (255, 255, 255), thickness)

traj_buffers = {}

frame_idx = 0
while frame_idx <= MAX_FRAME:
    ret, frame = cap.read()
    if not ret: break
    
    frame_gt = df_gt[df_gt["frame"] == frame_idx]
    for _, row in frame_gt.iterrows():
        box = [int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])]
        cls_name = row["class_name"]
        tid = int(row["track_id"])
        
        # Keep track of center for trajectories
        cx, cy = int(row["center_x"]), int(row["center_y"])
        if tid not in traj_buffers:
            traj_buffers[tid] = deque(maxlen=60) # 2 seconds of trail
        traj_buffers[tid].append((cx, cy))
        
        color = CLASS_COLORS.get(cls_name, (200, 200, 200))
        
        # Draw Trajectory
        pts = list(traj_buffers[tid])
        for k in range(1, len(pts)):
            alpha = k / len(pts)
            faded = tuple(int(c * alpha) for c in color)
            cv2.line(frame, pts[k-1], pts[k], faded, 2, cv2.LINE_AA)
            
        label = f"ID:{tid}"
        draw_box_label(frame, box, label, color)
        
    writer.write(frame)
    if frame_idx % 30 == 0:
        print(f"Rendered frame {frame_idx}/{MAX_FRAME}")
    frame_idx += 1

cap.release()
writer.release()
print(f"   -> Ground truth video saved to {OUT_VIDEO}")
print("\nAll tasks completed successfully!")

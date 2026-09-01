import os
from pathlib import Path
from typing import Tuple, List, Union, Optional
import numpy as np
import pandas as pd
import cv2

# Strict Calibration Constants
CENTER_X = 43.5
CENTER_Y = 28.5
R_INNER = 6.0
R_OUTER = 14.0
R_LANE_DIVIDER = 10.0


def detect_erratic_weaving(
    csv_or_df: Union[str, Path, pd.DataFrame],
    center_x: float = CENTER_X,
    center_y: float = CENTER_Y,
    r_min: float = R_INNER,
    r_max: float = R_OUTER,
    r_lane_divider: float = R_LANE_DIVIDER,
    window_frames: int = 90,
    min_transitions: int = 3,
    allowed_classes: Optional[List[str]] = None,
    fps: float = 30.0
) -> Tuple[pd.DataFrame, List[Union[int, str]]]:
    """
    Detects 'Erratic Lane Weaving' by tracking physical lane boundary crosses inside a roundabout.

    Dataset Schema:
    - Columns: ['frame', 'track_id', 'x', 'y', 'velocity_ms', 'vx', 'vy', 'class_name']

    Parameters:
    -----------
    csv_or_df : Union[str, Path, pd.DataFrame]
        Path to CSV file or existing DataFrame.
    """
    if allowed_classes is None:
        allowed_classes = ["car", "motorcycle", "bus", "truck", "van"]

    if isinstance(csv_or_df, (str, Path)):
        df = pd.read_csv(csv_or_df)
    else:
        df = csv_or_df

    if df.empty:
        df_out = df.copy()
        df_out["is_erratic_weaving"] = False
        return df_out, []

    df_out = df.copy()

    # Class filter: exclude pedestrians ('person') to match wrong_way_rule.py
    if "class_name" in df_out.columns:
        df_out = df_out[df_out["class_name"].isin(allowed_classes)].copy()

    # Metric coordinates setup with fallback, identical to wrong_way_rule.py
    if "world_x" in df_out.columns and "world_y" in df_out.columns:
        df_out["x_m"] = df_out["world_x"]
        df_out["y_m"] = df_out["world_y"]
    elif "x" in df_out.columns and "y" in df_out.columns:
        df_out["x_m"] = df_out["x"]
        df_out["y_m"] = df_out["y"]
    else:
        # Fallback if neither found
        df_out["x_m"] = df_out.get("world_x", df_out.get("x", 0.0))
        df_out["y_m"] = df_out.get("world_y", df_out.get("y", 0.0))

    if "vx" not in df_out.columns or "vy" not in df_out.columns:
        df_out = df_out.sort_values(["track_id", "frame"])
        dt = 1.0 / fps
        df_out["vx"] = (df_out.groupby("track_id")["x_m"].diff() / dt).bfill().ffill().fillna(0.0)
        df_out["vy"] = (df_out.groupby("track_id")["y_m"].diff() / dt).bfill().ffill().fillna(0.0)

    if "velocity_ms" not in df_out.columns:
        df_out["velocity_ms"] = np.hypot(df_out["vx"], df_out["vy"])

    # Ensure sorted by track_id and frame
    df_out = df_out.sort_values(["track_id", "frame"]).reset_index(drop=True)

    # 1. Calculate Radial Distance r using center_x and center_y
    dx = df_out["x_m"].values - center_x
    dy = df_out["y_m"].values - center_y
    r = np.hypot(dx, dy)

    # 2. Filter mask: inside roundabout bounds (r_min <= r <= r_max)
    in_ring = (r >= r_min) & (r <= r_max)

    # 3. Track positional state (1 if r >= r_lane_divider else 0)
    state = (r >= r_lane_divider).astype(int)
    df_out["_state"] = state

    # Detect state transitions (0->1 or 1->0)
    prev_state = df_out.groupby("track_id")["_state"].shift(1)
    has_prev = prev_state.notna()
    is_transition = (df_out["_state"] != prev_state) & has_prev & in_ring
    df_out["_transition"] = is_transition.astype(int)

    # 4. Group by track_id and apply rolling window of 90 frames
    rolling_transitions = (
        df_out.groupby("track_id")["_transition"]
        .transform(lambda s: s.rolling(window=window_frames, min_periods=1).sum())
    )

    # Trigger: Flag violation if transitions >= min_transitions inside roundabout bounds
    df_out["is_erratic_weaving"] = (rolling_transitions >= min_transitions) & in_ring

    # Extract unique violating track IDs
    unique_violators = df_out.loc[df_out["is_erratic_weaving"], "track_id"].unique().tolist()

    # Clean temporary internal columns
    df_out.drop(columns=["_state", "_transition"], inplace=True, errors="ignore")

    return df_out, unique_violators


# Alias for backward compatibility if needed
detect_jittering = detect_erratic_weaving


def visualize_erratic_weaving_video(
    df: pd.DataFrame,
    input_video_path: str,
    output_video_path: str,
    only_violation_frames: bool = False
):
    """
    Renders video annotations for Erratic Lane Weaving violations.
    """
    if not os.path.exists(input_video_path):
        print(f"Error: Video file not found at '{input_video_path}'")
        return

    print(f"\n--- Rendering Erratic Lane Weaving Video Visualization ---")
    print(f"Input Video: {input_video_path}")
    print(f"Output Video: {output_video_path}")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Failed to open video file: {input_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    tracks_by_frame = df.groupby("frame")
    frame_idx = 0
    written_count = 0

    MAGENTA = (255, 0, 255)
    GRAY = (180, 180, 180)

    while True:
        ret, frame_img = cap.read()
        if not ret:
            break

        if frame_idx in tracks_by_frame.groups:
            frame_rows = tracks_by_frame.get_group(frame_idx)
            has_violation = frame_rows["is_erratic_weaving"].any() if "is_erratic_weaving" in frame_rows.columns else False

            if not only_violation_frames or has_violation:
                for _, row in frame_rows.iterrows():
                    is_violating = bool(row.get("is_erratic_weaving", False))
                    tid = int(row["track_id"])

                    if {"x1", "y1", "x2", "y2"}.issubset(row.index) and not pd.isna(row["x1"]):
                        x1, y1 = int(row["x1"]), int(row["y1"])
                        x2, y2 = int(row["x2"]), int(row["y2"])
                    else:
                        cx = int(row.get("x", row.get("world_x", 0)))
                        cy = int(row.get("y", row.get("world_y", 0)))
                        w, h = 50, 50
                        x1, y1 = max(0, cx - w // 2), max(0, cy - h // 2)
                        x2, y2 = min(width, cx + w // 2), min(height, cy + h // 2)

                    if is_violating:
                        cv2.rectangle(frame_img, (x1, y1), (x2, y2), MAGENTA, 3)
                        banner_text = f"ID:{tid} [ERRATIC WEAVING]"
                        (tw, th), baseline = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        ty = max(y1 - 5, th + 5)
                        cv2.rectangle(frame_img, (x1, ty - th - 4), (x1 + tw + 6, ty + baseline + 2), MAGENTA, -1)
                        cv2.putText(frame_img, banner_text, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        cv2.rectangle(frame_img, (x1, y1), (x2, y2), GRAY, 1)
                        cv2.putText(frame_img, f"ID:{tid}", (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1, cv2.LINE_AA)

                hud = f"Frame: {frame_idx}/{total_frames} | Erratic Weaving Violations: {int(has_violation)}"
                cv2.putText(frame_img, hud, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

                out.write(frame_img)
                written_count += 1

        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"Processed {frame_idx}/{total_frames} video frames...")

    cap.release()
    out.release()
    print(f"Successfully saved Erratic Lane Weaving annotated video -> {output_video_path} ({written_count} frames written)\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Erratic Lane Weaving Violation Detection & Visualization")
    # Put your trajectory tracks CSV file path here:
    parser.add_argument("--tracks", type=str, default="data/tracks.csv", help="Path to input trajectory CSV file (e.g. data/tracks.csv)")
    # Put your video file path here (optional, for visualization):
    parser.add_argument("--video", type=str, default="data/intersection.mp4", help="Path to input video file (e.g. data/intersection.mp4)")
    # Put your output violations CSV path here:
    parser.add_argument("--output_csv", type=str, default="outputs/output_erratic_weaving.csv", help="Path to output violations CSV file")
    # Put your output annotated video path here:
    parser.add_argument("--output_video", type=str, default="outputs/video/output_erratic_weaving_vis.mp4", help="Path to output annotated video")

    args = parser.parse_args()

    # Resolve CSV file
    csv_file = args.tracks
    if not os.path.exists(csv_file):
        base_dir = os.path.dirname(__file__)
        fallback_cands = [
            os.path.join(base_dir, "data", "long1_tracks_narain_cleaned_edited.csv"),
            os.path.join(base_dir, "..", "..", "data", "long1_tracks_narain_cleaned_edited.csv"),
        ]
        for c in fallback_cands:
            if os.path.exists(c):
                csv_file = c
                break

    video_file = args.video
    output_csv = args.output_csv
    output_video = args.output_video

    if not os.path.exists(csv_file):
        print(f"File not found at '{csv_file}'. Please specify your tracks file via --tracks <path_to_csv>.")
    else:
        print(f"Processing Erratic Lane Weaving Detection on: {csv_file}")
        res_df, violators = detect_erratic_weaving(csv_file)
        
        print("\n--- Erratic Lane Weaving Summary ---")
        print(f"Total Rows Processed: {len(res_df)}")
        print(f"Erratic Weaving Flagged Frames: {res_df['is_erratic_weaving'].sum()}")
        print(f"Unique Violating Track IDs ({len(violators)}): {violators}")
        
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        res_df.to_csv(output_csv, index=False)
        print(f"Results saved to: {output_csv}")

        if os.path.exists(video_file):
            os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)
            visualize_erratic_weaving_video(res_df, video_file, output_video)
        else:
            print(f"\nNote: Video file '{video_file}' not found (skipping video generation).")

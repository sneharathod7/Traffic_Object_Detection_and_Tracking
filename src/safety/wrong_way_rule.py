import os
import argparse
import pandas as pd
import numpy as np


def detect_wrong_way_violations(csv_file=None, output_csv="outputs/wrong_way.csv"):
    if csv_file is None:
        base_dir = os.path.dirname(__file__)
        data_candidates = [
            os.path.join(base_dir, "data", "tracks.csv"),
            os.path.join(base_dir, "data", "long1_tracks_narain_cleaned_edited.csv"),
            os.path.join(base_dir, "..", "..", "data", "long1_tracks_narain_cleaned_edited.csv"),
            "data/tracks.csv",
            "data/long1_tracks_narain_cleaned_edited.csv",
        ]
        for candidate in data_candidates:
            if os.path.exists(candidate):
                csv_file = candidate
                break
        if csv_file is None:
            csv_file = "data/tracks.csv"

    # Load dataset
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found. Please specify via --tracks <path_to_csv>.")
        return None, {}

    try:
        from .calibration import CENTER_X as X_c, CENTER_Y as Y_c, R_INNER as R_MIN, R_OUTER as R_MAX
    except ImportError:
        try:
            from calibration import CENTER_X as X_c, CENTER_Y as Y_c, R_INNER as R_MIN, R_OUTER as R_MAX
        except ImportError:
            X_c, Y_c, R_MIN, R_MAX = 28.5, 43.4, 6.0, 14.0

    FPS = 30.0
    OMEGA_THRESHOLD = -0.1
    CONSECUTIVE_FRAMES_THRESHOLD = 45
    MIN_SPEED = 1.5
    ALLOWED_CLASSES = ["car", "motorcycle", "bus", "truck", "van"]

    # Filter out pedestrians ('person') who walk on sidewalks
    if "class_name" in df.columns:
        df = df[df["class_name"].isin(ALLOWED_CLASSES)].copy()

    # Metric coordinates setup
    if "world_x" in df.columns and "world_y" in df.columns:
        df["x_m"] = df["world_x"]
        df["y_m"] = df["world_y"]
    else:
        df["x_m"] = df["x"]
        df["y_m"] = df["y"]

    df = df.sort_values(by=["track_id", "frame"]).reset_index(drop=True)

    # 1. Vectorized Polar Coordinates
    dx = df["x_m"].values - X_c
    dy = df["y_m"].values - Y_c
    df["r"] = np.hypot(dx, dy)
    df["theta"] = np.arctan2(dy, dx)

    # 2. Vectorized Velocity
    if "velocity_ms" not in df.columns:
        dt = 1.0 / FPS
        vx = (df.groupby("track_id")["x_m"].diff() / dt).bfill().ffill().fillna(0.0)
        vy = (df.groupby("track_id")["y_m"].diff() / dt).bfill().ffill().fillna(0.0)
        df["velocity_ms"] = np.hypot(vx, vy)

    # 3. Shortest angular change & angular velocity (Vectorized)
    theta_shift = df.groupby("track_id")["theta"].shift(1)
    delta_theta = np.arctan2(np.sin(df["theta"] - theta_shift), np.cos(df["theta"] - theta_shift))
    df["omega"] = delta_theta * FPS

    # 4. Ring, Speed & Direction Filter (Vectorized)
    df["is_in_ring"] = (df["r"] >= R_MIN) & (df["r"] <= R_MAX)
    df["is_moving"] = df["velocity_ms"] >= MIN_SPEED
    df["is_wrong_way"] = (df["omega"] < OMEGA_THRESHOLD) & df["is_in_ring"] & df["is_moving"]

    # 5. Consecutive block tracking (Vectorized)
    df["consecutive_group"] = (df["is_wrong_way"] != df.groupby("track_id")["is_wrong_way"].shift(1)).cumsum()
    
    wrong_df = df[df["is_wrong_way"]]
    if wrong_df.empty:
        print("No Wrong-Way Driving Violations detected.")
        return None, {}

    counts = wrong_df.groupby(["track_id", "consecutive_group"])["frame"].transform("count")
    valid_wrong = wrong_df[counts >= CONSECUTIVE_FRAMES_THRESHOLD]

    if valid_wrong.empty:
        print("No Wrong-Way Driving Violations detected.")
        return None, {}

    violations = {}
    for tid, grp in valid_wrong.groupby("track_id"):
        first_row = grp.iloc[0]
        c_name = first_row["class_name"] if "class_name" in first_row.index else "vehicle"
        violations[tid] = {
            "class_name": c_name,
            "start_frame": int(first_row["frame"])
        }

    print("--- Calibrated Wrong-Way Driving Violations Summary ---")
    class_counts = {}
    for vid, info in violations.items():
        c_name = info["class_name"]
        class_counts[c_name] = class_counts.get(c_name, 0) + 1
        
    print("\nTotal number of unique track IDs flagged by class:")
    for c_name, count in class_counts.items():
        print(f"- {c_name}: {count}")

    print("\nSpecific track IDs caught violating the rule:")
    for track_id, info in violations.items():
        print(f"Track ID {track_id} (Class: {info['class_name']}) - Violation started at frame: {info['start_frame']}")

    # Save output wrong_way.csv for visualization consumption
    v_rows = [{"track_id": tid, "class_name": info["class_name"], "start_frame": info["start_frame"]} for tid, info in violations.items()]
    out_df = pd.DataFrame(v_rows)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        out_df.to_csv(output_csv, index=False)
        print(f"\nSaved wrong-way violations to '{output_csv}' ({len(out_df)} flagged vehicles).")

    return out_df, violations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wrong-Way Driving Rule Violation Detection")
    # Put your input trajectory CSV path here:
    parser.add_argument("--tracks", type=str, default="data/tracks.csv", help="Path to input trajectory CSV file (e.g. data/tracks.csv)")
    # Put your output violations CSV path here:
    parser.add_argument("--output", type=str, default="outputs/wrong_way.csv", help="Path to output violations CSV file")

    args = parser.parse_args()
    detect_wrong_way_violations(csv_file=args.tracks, output_csv=args.output)

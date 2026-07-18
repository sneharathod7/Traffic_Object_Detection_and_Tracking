import pandas as pd
import numpy as np
from pathlib import Path

try:
    from .calibration import CENTER_X as X_C, CENTER_Y as Y_C, R_INNER, R_OUTER
except ImportError:
    from calibration import CENTER_X as X_C, CENTER_Y as Y_C, R_INNER, R_OUTER
CONGESTION_THRESHOLD = 2
MIN_FRAMES = 6

STRAIGHT_MOVEMENTS = {
    ("NORTH", "SOUTH"),
    ("SOUTH", "NORTH"),
    ("EAST", "WEST"),
    ("WEST", "EAST"),
}

TURNING_MOVEMENTS = {
    ("EAST", "NORTH"),
    ("EAST", "SOUTH"),
    ("WEST", "NORTH"),
    ("WEST", "SOUTH"),
    ("NORTH", "EAST"),
    ("NORTH", "WEST"),
    ("SOUTH", "EAST"),
    ("SOUTH", "WEST"),
}


def determine_direction(dx: float, dy: float) -> str:
    if abs(dx) >= abs(dy):
        return "EAST" if dx > 0 else "WEST"
    return "SOUTH" if dy > 0 else "NORTH"


def _is_turning(entry_direction: str, exit_direction: str) -> bool:
    return (entry_direction, exit_direction) in TURNING_MOVEMENTS


def detect_unsafe_roundabout_shortcuts(csv_file: str, output_csv_path: str | Path | None = None):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        return

    required_columns = {"track_id", "frame", "world_x", "world_y", "class_name"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns in input CSV: {sorted(missing)}")

    if output_csv_path is None:
        output_csv_path = Path(__file__).resolve().parent / "csv_outputs" / "unsafe_shortcut_violations.csv"
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["dx"] = df["world_x"] - X_C
    df["dy"] = df["world_y"] - Y_C
    df["r"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)

    frame_groups = {frame: group for frame, group in df.groupby("frame")}
    violations = []

    for track_id, track in df.groupby("track_id"):
        track = track.sort_values("frame")
        if len(track) < MIN_FRAMES:
            continue

        first = track.iloc[0]
        last = track.iloc[-1]
        entry_direction = determine_direction(first["dx"], first["dy"])
        exit_direction = determine_direction(last["dx"], last["dy"])

        is_straight = (entry_direction, exit_direction) in STRAIGHT_MOVEMENTS
        is_turning = _is_turning(entry_direction, exit_direction)

        if not (is_straight or is_turning):
            continue

        # Radial band check: must enter the roundabout approach zone
        r_min = track["r"].min()
        if r_min >= R_OUTER:
            continue

        # Angular traversal check
        theta = np.arctan2(track["dy"], track["dx"])
        theta_unwrapped = np.unwrap(theta)
        #check for th4 unwrappng
        total_angular_change = np.abs(theta_unwrapped[-1] - theta_unwrapped[0]) * 180.0 / np.pi

        is_shortcut = False
        
        # Based on the specific intersection geometry and user-provided image,
        # the ONLY physical shortcuts are wrong-way right turns that cut the corner
        # (North to West, and South to East).
        # Proper paths (clockwise around the island) have angular change ~270 degrees (> 150)
        # Shortcut paths (cutting the corner) have angular change ~90 degrees (< 150)
        shortcut_paths = {("NORTH", "WEST"), ("SOUTH", "EAST")}
        
        if (entry_direction, exit_direction) in shortcut_paths:
            if total_angular_change < 150.0:
                is_shortcut = True

        if not is_shortcut:
            continue

        conflict_frames = []
        for _, row in track.iterrows():
            same_frame = frame_groups[row["frame"]]
            other_vehicles = same_frame[same_frame["track_id"] != track_id]
            other_in_outer = (other_vehicles["r"] < R_OUTER).sum()
            if other_in_outer >= CONGESTION_THRESHOLD:
                conflict_frames.append(int(row["frame"]))

        if not conflict_frames:
            continue

        violation_start_frame = min(conflict_frames)
        class_name = str(track["class_name"].iloc[0])
        reason = (
            f"{entry_direction} to {exit_direction} shortcut without roundabout traversal during congestion"
        )

        violations.append({
            "track_id": int(track_id),
            "class_name": class_name,
            "entry_direction": entry_direction,
            "exit_direction": exit_direction,
            "violation_start_frame": int(violation_start_frame),
            "violation_type": "Unsafe Roundabout Shortcut",
            "reason": reason,
        })

    output_df = pd.DataFrame(
        violations,
        columns=[
            "track_id",
            "class_name",
            "entry_direction",
            "exit_direction",
            "violation_start_frame",
            "violation_type",
            "reason",
        ],
    )
    output_df.to_csv(output_csv_path, index=False)
    print(f"Saved {len(output_df)} unsafe roundabout shortcut violation(s) -> {output_csv_path}")
    return output_csv_path


if __name__ == "__main__":
    import os
    default_tracks = r"D:\btp\narain_data\full1_tracks (1).csv"
    if not os.path.exists(default_tracks):
        possible_path = os.path.join(os.path.dirname(__file__), "..", "..", "narain_data", "full1_tracks (1).csv")
        if os.path.exists(possible_path):
            default_tracks = possible_path
            
    detect_unsafe_roundabout_shortcuts(default_tracks)

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

try:
    from .calibration import CENTER_X as X_C, CENTER_Y as Y_C, R_OUTER
except ImportError:
    from calibration import CENTER_X as X_C, CENTER_Y as Y_C, R_OUTER
MIN_DIRECTION_ANGLE_DEG = 30.0
BEHIND_AHEAD_THRESHOLD = 0.5
OVERTAKE_PERSISTENCE_FRAMES = 3
LATERAL_MOVEMENT_THRESHOLD = 0.8


def _angle_between(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 < 1e-6 or mag2 < 1e-6:
        return 180.0
    cos = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos))


def _normalize(v: Tuple[float, float]) -> Tuple[float, float]:
    mag = math.hypot(v[0], v[1])
    if mag < 1e-6:
        return 0.0, 0.0
    return v[0] / mag, v[1] / mag


def _cross(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _track_direction(track: pd.DataFrame) -> Optional[Tuple[float, float]]:
    if len(track) < 2:
        return None
    start = track.iloc[0]
    end = track.iloc[-1]
    direction = (float(end["world_x"]) - float(start["world_x"]),
                 float(end["world_y"]) - float(start["world_y"]))
    if math.hypot(direction[0], direction[1]) < 0.1:
        return None
    return _normalize(direction)


def _load_data(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    if "velocity_ms" not in df.columns and "velocity" in df.columns:
        df = df.rename(columns={"velocity": "velocity_ms"})
    required = {"track_id", "frame", "world_x", "world_y", "class_name"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")
    df = df.copy()

    df["dx"] = df["world_x"] - X_C
    df["dy"] = df["world_y"] - Y_C
    df["r"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    return df


def detect_unsafe_overtaking(csv_file: str, output_csv_path: Union[str, Path, None] = None):
    df = _load_data(csv_file)

    if output_csv_path is None:
        output_csv_path = Path(__file__).resolve().parent / "csv_outputs" / "unsafe_overtaking_violations.csv"
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    tracks = {tid: group.sort_values("frame") for tid, group in df.groupby("track_id")}
    tracks_indexed = {tid: t.set_index("frame") for tid, t in tracks.items()}
    directions: Dict[int, Tuple[float, float]] = {}
    for tid, track in tracks.items():
        heading = _track_direction(track)
        if heading is not None:
            directions[tid] = heading

    frame_index: Dict[int, List[int]] = {}
    for tid, track in tracks.items():
        for frame in track["frame"]:
            frame_index.setdefault(int(frame), []).append(tid)

    violations = []
    seen_pairs = set()

    for track_id, track in tracks.items():
        if track_id not in directions:
            continue
        dir_a = directions[track_id]

        for frame in sorted(track["frame"].unique()):
            other_ids = frame_index.get(int(frame), [])
            for oth_id in other_ids:
                if oth_id == track_id or (track_id, oth_id) in seen_pairs:
                    continue
                seen_pairs.add((track_id, oth_id))
                seen_pairs.add((oth_id, track_id))
                if oth_id not in directions:
                    continue
                dir_b = directions[oth_id]
                if _angle_between(dir_a, dir_b) > MIN_DIRECTION_ANGLE_DEG:
                    continue

                other = tracks[oth_id]
                overlap_frames = sorted(set(track["frame"]).intersection(other["frame"]))
                if len(overlap_frames) < OVERTAKE_PERSISTENCE_FRAMES:
                    continue

                rel_dir = _normalize(((dir_a[0] + dir_b[0]) / 2.0, (dir_a[1] + dir_b[1]) / 2.0))
                if rel_dir == (0.0, 0.0):
                    continue

                track_rows = tracks_indexed[track_id]
                other_rows = tracks_indexed[oth_id]

                sequence = []
                for frame_id in overlap_frames:
                    row_a = track_rows.loc[frame_id]
                    row_b = other_rows.loc[frame_id]
                    rel = (float(row_a["world_x"]) - float(row_b["world_x"]),
                           float(row_a["world_y"]) - float(row_b["world_y"]))
                    forward = _dot(rel, rel_dir)
                    lateral = _cross(rel_dir, rel)
                    r_zone = float(row_a["r"]) < R_OUTER and float(row_b["r"]) < R_OUTER
                    sequence.append({
                        "frame": int(frame_id),
                        "forward": forward,
                        "lateral": lateral,
                        "dist": math.hypot(rel[0], rel[1]),
                        "r_zone": r_zone,
                    })

                if len(sequence) < OVERTAKE_PERSISTENCE_FRAMES:
                    continue

                behind_frames = [s for s in sequence if s["forward"] < -BEHIND_AHEAD_THRESHOLD]
                ahead_frames = [s for s in sequence if s["forward"] > BEHIND_AHEAD_THRESHOLD]
                if not behind_frames or not ahead_frames:
                    continue

                first_behind = behind_frames[0]["frame"]
                last_ahead = ahead_frames[-1]["frame"]
                if first_behind >= last_ahead:
                    continue

                crossing_frames = [s for s in sequence if abs(s["forward"]) <= BEHIND_AHEAD_THRESHOLD]
                if not crossing_frames:
                    continue

                # Ensure the passage persists over multiple frames
                crossing_frame = crossing_frames[0]["frame"]
                window = [s for s in sequence if first_behind <= s["frame"] <= last_ahead]
                if len(window) < OVERTAKE_PERSISTENCE_FRAMES:
                    continue

                # Distance should shrink as A approaches B and then can increase after pass
                pre_dist = [s["dist"] for s in window if s["frame"] <= crossing_frame]
                post_dist = [s["dist"] for s in window if s["frame"] >= crossing_frame]
                if not pre_dist or not post_dist or min(pre_dist) >= max(post_dist):
                    continue

                lateral_before = [abs(s["lateral"]) for s in window if s["frame"] <= crossing_frame]
                lateral_after = [abs(s["lateral"]) for s in window if s["frame"] >= crossing_frame]
                if not lateral_before or not lateral_after:
                    continue
                lateral_change = abs(np.median(lateral_after) - np.median(lateral_before))
                if lateral_change < LATERAL_MOVEMENT_THRESHOLD:
                    continue

                if not any(s["r_zone"] for s in window):
                    continue

                start_frame = first_behind
                reason = (f"Vehicle {track_id} overtook {oth_id} inside restricted roundabout zone "
                          f"after approaching from the same direction")
                location = "restricted zone"

                violations.append({
                    "track_id": int(track_id),
                    "overtaken_vehicle_id": int(oth_id),
                    "class_name": str(track["class_name"].iloc[0]),
                    "start_frame": int(start_frame),
                    "location": location,
                    "violation_type": "Unsafe Overtaking",
                    "reason": reason,
                })
                seen_pairs.add((track_id, oth_id))
                seen_pairs.add((oth_id, track_id))
                break

    output_df = pd.DataFrame(
        violations,
        columns=[
            "track_id",
            "overtaken_vehicle_id",
            "class_name",
            "start_frame",
            "location",
            "violation_type",
            "reason",
        ],
    )
    output_df.to_csv(output_csv_path, index=False)
    print(f"Saved {len(output_df)} unsafe overtaking violation(s) -> {output_csv_path}")
    return output_csv_path


if __name__ == "__main__":
    import os
    default_tracks = r"D:\btp\narain_data\full1_tracks (1).csv"
    if not os.path.exists(default_tracks):
        possible_path = os.path.join(os.path.dirname(__file__), "..", "..", "narain_data", "full1_tracks (1).csv")
        if os.path.exists(possible_path):
            default_tracks = possible_path
            
    detect_unsafe_overtaking(str(default_tracks))

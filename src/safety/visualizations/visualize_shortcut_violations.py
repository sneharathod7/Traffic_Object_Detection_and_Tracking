import cv2
import pandas as pd
from pathlib import Path


def visualize_shortcut_violations(
    video_path: str,
    tracks_csv_path: str,
    shortcut_csv_path: str,
    output_video_path: str,
):
    print(f"Loading tracks from: {tracks_csv_path}")
    tracks_df = pd.read_csv(tracks_csv_path)

    print(f"Loading shortcut violations from: {shortcut_csv_path}")
    shortcuts_df = pd.read_csv(shortcut_csv_path)
    if "track_id" not in shortcuts_df.columns:
        print("Error: shortcut CSV must contain 'track_id'")
        return

    shortcut_start_frames = {
        int(row["track_id"]): int(row["violation_start_frame"])
        for _, row in shortcuts_df.dropna(subset=["track_id", "violation_start_frame"]).iterrows()
    }
    if not shortcut_start_frames:
        print("No unsafe shortcut violations found in shortcut CSV.")
        return

    print("Grouping track data by frame...")
    frames_group = tracks_df.groupby("frame")

    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Generating output video: {output_video_path}")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frames_group.groups:
            frame_data = frames_group.get_group(frame_idx)
            for _, track in frame_data.iterrows():
                tid = int(track["track_id"])
                if tid not in shortcut_start_frames:
                    continue
                if frame_idx < shortcut_start_frames[tid]:
                    continue

                x1 = int(track["x1"])
                y1 = int(track["y1"])
                x2 = int(track["x2"])
                y2 = int(track["y2"])
                color = (0, 165, 255)  # orange
                label_str = f"ID:{tid} Shortcut"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label_str, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        out.write(frame)
        if frame_idx % 100 == 0:
            print(f"Processed frame {frame_idx}/{total_frames}")
        frame_idx += 1

    cap.release()
    out.release()
    print("Done generating unsafe shortcut visualization!")


if __name__ == "__main__":
    video_file = r"D:\btp\narain_data\full1 (1).MP4"
    tracks_file = r"D:\btp\narain_data\full1_tracks (1).csv"
    shortcut_file = r"D:\btp\Traffic_Object_Detection_and_Tracking\src\safety\csv_outputs\unsafe_shortcut_violations.csv"
    out_file = r"D:\btp\Traffic_Object_Detection_and_Tracking\outputs\video\full1_unsafe_shortcut.mp4"
    visualize_shortcut_violations(video_file, tracks_file, shortcut_file, out_file)

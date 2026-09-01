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
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Unsafe Roundabout Shortcut Violations")
    # Put your video file path here:
    parser.add_argument("--video", type=str, default="data/intersection.mp4", help="Path to input video file (e.g. data/intersection.mp4)")
    # Put your tracks CSV file path here:
    parser.add_argument("--tracks", type=str, default="data/tracks.csv", help="Path to input trajectory CSV file (e.g. data/tracks.csv)")
    # Put your shortcut violations CSV path here:
    parser.add_argument("--shortcuts", type=str, default="outputs/unsafe_shortcut_violations.csv", help="Path to shortcut violations CSV")
    # Put your output annotated video path here:
    parser.add_argument("--output", type=str, default="outputs/video/unsafe_shortcut_annotated.mp4", help="Path to output annotated MP4 video")
    
    args = parser.parse_args()
    visualize_shortcut_violations(args.video, args.tracks, args.shortcuts, args.output)

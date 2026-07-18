import cv2
import pandas as pd
import numpy as np
from pathlib import Path

def visualize_wrong_way(
    video_path,
    tracks_csv_path,
    wrong_way_csv_path,
    output_video_path
):
    print(f"Loading tracks from: {tracks_csv_path}")
    tracks_df = pd.read_csv(tracks_csv_path)
    
    print(f"Loading wrong way violations from: {wrong_way_csv_path}")
    try:
        ww_df = pd.read_csv(wrong_way_csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{wrong_way_csv_path}' was not found.")
        return

    if 'track_id' not in ww_df.columns:
        print(f"Error: The wrong-way CSV file must contain a 'track_id' column.")
        return

    # 1. Parse wrong way tracks
    if 'start_frame' in ww_df.columns:
        wrong_way_start_frames = {
            int(row['track_id']): int(row['start_frame'])
            for _, row in ww_df.dropna(subset=['track_id', 'start_frame']).iterrows()
        }
    else:
        wrong_way_start_frames = {
            int(track_id): 0
            for track_id in ww_df['track_id'].dropna().astype(int)
        }

    if not wrong_way_start_frames:
        print("No wrong-way violations found in the CSV file.")
        return
    
    # Pre-group tracks by frame for faster lookup
    print("Grouping track data by frame...")
    frames_group = tracks_df.groupby('frame')
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Prepare VideoWriter
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Generating output video: {output_video_path}")
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw bounding boxes if there are any for this frame
        if frame_idx in frames_group.groups:
            frame_data = frames_group.get_group(frame_idx)
            
            for _, track in frame_data.iterrows():
                tid = int(track['track_id'])
                
                if tid not in wrong_way_start_frames:
                    continue

                start_frame = wrong_way_start_frames[tid]
                if frame_idx < start_frame:
                    continue

                x1, y1, x2, y2 = int(track['x1']), int(track['y1']), int(track['x2']), int(track['y2'])
                
                # Magenta/Purple color for Wrong-Way
                color = (255, 0, 255)
                label_str = f"ID:{tid} Wrong-Way"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label_str, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
        out.write(frame)
        
        if frame_idx % 100 == 0:
            print(f"Processed frame {frame_idx}/{total_frames}")
            
        frame_idx += 1
        
    cap.release()
    out.release()
    print("Done generating wrong-way visualization!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize wrong-way driving violations on video")
    parser.add_argument("--video", type=str, default=r"D:\btp\narain_data\full1 (1).MP4")
    parser.add_argument("--tracks", type=str, default=r"D:\btp\narain_data\full1_tracks (1).csv")
    parser.add_argument("--violations", type=str, default=r"D:\btp\Traffic_Object_Detection_and_Tracking\src\safety\csv_outputs\wrong_way.csv")
    parser.add_argument("--output", type=str, default=r"D:\btp\Traffic_Object_Detection_and_Tracking\outputs\video\full1_wrong_way_annotated.mp4")
    args = parser.parse_args()
    
    visualize_wrong_way(args.video, args.tracks, args.violations, args.output)

import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import math

def _resolve(path_str: str) -> str:
    if not path_str or os.path.exists(path_str):
        return path_str
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    cands = [
        os.path.join(project_root, path_str),
        os.path.join(project_root, "newsafety_rules", path_str),
        os.path.join(project_root, "newsafety_rules", "data", os.path.basename(path_str)),
        os.path.join(project_root, "data", os.path.basename(path_str)),
        os.path.join(project_root, "src", "safety", os.path.basename(path_str)),
        os.path.join(script_dir, "..", path_str),
    ]
    for c in cands:
        if os.path.exists(c):
            return os.path.abspath(c)
    return path_str

def visualize_violations(
    video_path,
    tracks_csv_path,
    rule_csv_path,
    output_video_path
):
    video_path = _resolve(video_path)
    tracks_csv_path = _resolve(tracks_csv_path)
    rule_csv_path = _resolve(rule_csv_path)

    print(f"Loading tracks from: {tracks_csv_path}")
    tracks_df = pd.read_csv(tracks_csv_path)
    
    print(f"Loading rules from: {rule_csv_path}")
    rules_df = pd.read_csv(rule_csv_path)

    # Parse tailgating frames
    tailgating_df = rules_df[rules_df['violation_type'] == 'Tailgating']
    tailgating_dict = {}  # frame -> list of (follower_id, leader_id)
    
    for _, row in tailgating_df.iterrows():
        if pd.isna(row['frame']):
            continue
        frame = int(row['frame'])
        follower_id = int(row['track_id'])
        leader_id = int(row['leader_track_id']) if not pd.isna(row['leader_track_id']) else -1
        
        if frame not in tailgating_dict:
            tailgating_dict[frame] = []
        tailgating_dict[frame].append((follower_id, leader_id))
    
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
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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
            
            tg_data = tailgating_dict.get(frame_idx, [])
            tg_followers = {item[0] for item in tg_data}
            tg_leaders = {item[1] for item in tg_data}
            
            # Map of track_id to center point and bounding box for connecting lines
            pos_map = {}
            
            for _, track in frame_data.iterrows():
                tid = int(track['track_id'])
                
                is_follower = tid in tg_followers
                is_leader = tid in tg_leaders
                
                if not (is_follower or is_leader):
                    continue
                    
                if {'x1', 'y1', 'x2', 'y2'}.issubset(track.index) and not pd.isna(track['x1']):
                    x1, y1, x2, y2 = int(track['x1']), int(track['y1']), int(track['x2']), int(track['y2'])
                else:
                    cx = int(track.get('x', track.get('world_x', 0)))
                    cy = int(track.get('y', track.get('world_y', 0)))
                    w, h = 50, 50
                    x1, y1 = max(0, cx - w // 2), max(0, cy - h // 2)
                    x2, y2 = min(width, cx + w // 2), min(height, cy + h // 2)
                
                center_pt = ((x1 + x2) // 2, (y1 + y2) // 2)
                pos_map[tid] = (x1, y1, x2, y2, center_pt)

            # Draw connection lines between followers and leaders first (so they are under the boxes)
            for f_id, l_id in tg_data:
                if f_id in pos_map and l_id in pos_map:
                    pt_follower = pos_map[f_id][4]
                    pt_leader = pos_map[l_id][4]
                    # Draw dotted line or solid line to represent tailgating interaction
                    cv2.line(frame, pt_follower, pt_leader, (0, 0, 255), 2, cv2.LINE_AA)

            # Now draw bounding boxes and labels
            for tid, (x1, y1, x2, y2, center_pt) in pos_map.items():
                is_follower = tid in tg_followers
                
                if is_follower:
                    color = (0, 165, 255) # Orange (BGR) for follower
                    label_str = f"Follower ID:{tid} [TG]"
                else:
                    color = (0, 255, 255) # Yellow (BGR) for leader
                    label_str = f"Leader ID:{tid}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label_str, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                
        out.write(frame)
        
        if frame_idx % 100 == 0:
            print(f"Processed frame {frame_idx}/{total_frames}")
            
        frame_idx += 1
        
    cap.release()
    out.release()
    print("Done generating visualization!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Tailgating / Safe-Space Violations")
    # Put your video file path here:
    parser.add_argument("--video", type=str, default="data/intersection.mp4", help="Path to input video file (e.g. data/intersection.mp4)")
    # Put your trajectory tracks CSV path here:
    parser.add_argument("--tracks", type=str, default="data/tracks.csv", help="Path to trajectory tracks CSV file (e.g. data/tracks.csv)")
    # Put your tailgating rule violations CSV path here:
    parser.add_argument("--rule", type=str, default="outputs/tailgating_violations.csv", help="Path to violations CSV file (e.g. outputs/tailgating_violations.csv)")
    # Put your output annotated video path here:
    parser.add_argument("--output", type=str, default="outputs/video/tailgating_annotated.mp4", help="Path to output annotated video")
    
    args = parser.parse_args()
    visualize_violations(args.video, args.tracks, args.rule, args.output)
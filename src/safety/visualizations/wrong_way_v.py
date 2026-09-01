# import os
# import cv2
# import pandas as pd
# import numpy as np
# from pathlib import Path


# def resolve_file_path(path_str: str) -> str:
#     """
#     Attempts to resolve candidate relative and absolute paths across the project workspace.
#     """
#     if not path_str:
#         return path_str

#     if os.path.exists(path_str):
#         return path_str

#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

#     candidates = [
#         os.path.join(project_root, path_str),
#         os.path.join(project_root, "newsafety_rules", path_str),
#         os.path.join(project_root, "newsafety_rules", "data", os.path.basename(path_str)),
#         os.path.join(project_root, "src", "safety", path_str),
#         os.path.join(project_root, "src", "safety", "csv_outputs", os.path.basename(path_str)),
#         os.path.join(script_dir, "..", path_str),
#         os.path.join(script_dir, "..", "data", os.path.basename(path_str)),
#     ]

#     for cand in candidates:
#         if os.path.exists(cand):
#             return os.path.abspath(cand)

#     return path_str


# def visualize_wrong_way(
#     video_path: str,
#     tracks_csv_path: str,
#     wrong_way_csv_path: str,
#     output_video_path: str,
#     only_violation_frames: bool = False
# ):
#     resolved_tracks = resolve_file_path(tracks_csv_path)
#     resolved_video = resolve_file_path(video_path)
#     resolved_violations = resolve_file_path(wrong_way_csv_path)

#     print(f"Loading tracks from: {resolved_tracks}")
#     if not os.path.exists(resolved_tracks):
#         print(f"Error: The tracks file '{tracks_csv_path}' (resolved: '{resolved_tracks}') was not found.")
#         return

#     tracks_df = pd.read_csv(resolved_tracks)
    
#     print(f"Loading wrong way violations from: {resolved_violations}")
#     if not os.path.exists(resolved_violations):
#         print(f"Error: The wrong-way CSV file '{wrong_way_csv_path}' (resolved: '{resolved_violations}') was not found.")
#         return

#     ww_df = pd.read_csv(resolved_violations)

#     # CRITICAL FIX: Filter ONLY rows where is_wrong_way is True
#     if 'is_wrong_way' in ww_df.columns:
#         ww_df = ww_df[ww_df['is_wrong_way'] == True].copy()

#     if ww_df.empty:
#         print("No wrong-way driving violations found in the violations file.")
#         return

#     if 'track_id' not in ww_df.columns:
#         print(f"Error: The wrong-way CSV file must contain a 'track_id' column.")
#         return

#     # Create frame-level lookup set for active violation frames: (frame, track_id)
#     if 'frame' in ww_df.columns and 'track_id' in ww_df.columns:
#         active_violation_pairs = set(zip(ww_df['frame'].astype(int), ww_df['track_id'].astype(int)))
#     else:
#         # Fallback if only summary start_frame is present
#         wrong_way_start_frames = {}
#         for _, row in ww_df.iterrows():
#             tid = int(row['track_id'])
#             sf = int(row.get('start_frame', 0))
#             wrong_way_start_frames[tid] = sf
#         active_violation_pairs = None

#     violating_track_ids = set(ww_df['track_id'].astype(int).unique())
#     print(f"Unique Wrong-Way Violating Track IDs ({len(violating_track_ids)}): {sorted(list(violating_track_ids))}")

#     print("Grouping track data by frame...")
#     frames_group = tracks_df.groupby('frame')
    
#     print(f"Opening video: {resolved_video}")
#     if not os.path.exists(resolved_video):
#         print(f"Error: Video file '{resolved_video}' not found.")
#         return

#     cap = cv2.VideoCapture(resolved_video)
#     if not cap.isOpened():
#         print(f"Error opening video file {resolved_video}")
#         return
        
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     # Prepare VideoWriter
#     Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
#     print(f"Generating output video: {output_video_path}")
#     frame_idx = 0
#     written_count = 0
    
#     MAGENTA = (255, 0, 255)
#     GRAY = (180, 180, 180)

#     while True:
#         ret, frame_img = cap.read()
#         if not ret:
#             break
            
#         if frame_idx in frames_group.groups:
#             frame_data = frames_group.get_group(frame_idx)
            
#             # Check if any violation is occurring in this frame
#             has_violation = False
#             for _, track in frame_data.iterrows():
#                 tid = int(track['track_id'])
#                 if active_violation_pairs is not None:
#                     if (frame_idx, tid) in active_violation_pairs:
#                         has_violation = True
#                         break
#                 elif tid in violating_track_ids:
#                     if frame_idx >= wrong_way_start_frames.get(tid, 0):
#                         has_violation = True
#                         break

#             if not only_violation_frames or has_violation:
#                 for _, track in frame_data.iterrows():
#                     tid = int(track['track_id'])

#                     # Check if this specific vehicle is actively violating in this frame
#                     if active_violation_pairs is not None:
#                         is_violating = (frame_idx, tid) in active_violation_pairs
#                     else:
#                         is_violating = (tid in violating_track_ids) and (frame_idx >= wrong_way_start_frames.get(tid, 0))

#                     if {'x1', 'y1', 'x2', 'y2'}.issubset(track.index) and not pd.isna(track['x1']):
#                         x1, y1, x2, y2 = int(track['x1']), int(track['y1']), int(track['x2']), int(track['y2'])
#                     else:
#                         cx = int(track.get('x', track.get('world_x', 0)))
#                         cy = int(track.get('y', track.get('world_y', 0)))
#                         w, h = 50, 50
#                         x1, y1 = max(0, cx - w // 2), max(0, cy - h // 2)
#                         x2, y2 = min(width, cx + w // 2), min(height, cy + h // 2)
                    
#                     if is_violating:
#                         cv2.rectangle(frame_img, (x1, y1), (x2, y2), MAGENTA, 3)
#                         label_str = f"ID:{tid} Wrong-Way"
#                         (tw, th), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#                         ty = max(y1 - 5, th + 5)
#                         cv2.rectangle(frame_img, (x1, ty - th - 4), (x1 + tw + 6, ty + baseline + 2), MAGENTA, -1)
#                         cv2.putText(frame_img, label_str, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
#                     else:
#                         cv2.rectangle(frame_img, (x1, y1), (x2, y2), GRAY, 1)
#                         cv2.putText(frame_img, f"ID:{tid}", (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1, cv2.LINE_AA)

#                 hud = f"Frame: {frame_idx}/{total_frames} | Wrong-Way Violations: {int(has_violation)}"
#                 cv2.putText(frame_img, hud, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

#                 out.write(frame_img)
#                 written_count += 1
        
#         if frame_idx % 200 == 0:
#             print(f"Processed frame {frame_idx}/{total_frames}")
            
#         frame_idx += 1
        
#     cap.release()
#     out.release()
#     print(f"Done generating wrong-way visualization -> {output_video_path} ({written_count} frames written)")


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Visualize wrong-way driving violations on video")
#     parser.add_argument("--video", type=str, default=r"data\intersection.mp4")
#     parser.add_argument("--tracks", type=str, default=r"data\long1_tracks_narain_cleaned_edited.csv")
#     parser.add_argument("--violations", type=str, default=r"output_wrong_way.csv")
#     parser.add_argument("--output", type=str, default=r"outputs\video\wrong_way_annotated.mp4")
#     args = parser.parse_args()
    
#     visualize_wrong_way(args.video, args.tracks, args.violations, args.output)

import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path


def _resolve(path_str: str) -> str:
    if not path_str or os.path.exists(path_str):
        return path_str
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    cands = [
        os.path.join(project_root, path_str),
        os.path.join(project_root, "newsafety_rules", path_str),
        os.path.join(project_root, "newsafety_rules", "data", os.path.basename(path_str)),
        os.path.join(project_root, "newsafety_rules", "output_wrong_way.csv"),
        os.path.join(project_root, "src", "safety", "csv_outputs", os.path.basename(path_str)),
        os.path.join(project_root, "data", os.path.basename(path_str)),
        os.path.join(script_dir, "..", path_str),
    ]
    for c in cands:
        if os.path.exists(c):
            return os.path.abspath(c)
    return path_str


def visualize_wrong_way(
    video_path,
    tracks_csv_path,
    wrong_way_csv_path,
    output_video_path
):
    video_path = _resolve(video_path)
    tracks_csv_path = _resolve(tracks_csv_path)
    wrong_way_csv_path = _resolve(wrong_way_csv_path)

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

                if {'x1', 'y1', 'x2', 'y2'}.issubset(track.index) and not pd.isna(track['x1']):
                    x1, y1, x2, y2 = int(track['x1']), int(track['y1']), int(track['x2']), int(track['y2'])
                else:
                    cx = int(track.get('x', track.get('world_x', 0)))
                    cy = int(track.get('y', track.get('world_y', 0)))
                    w, h = 50, 50
                    x1, y1 = max(0, cx - w // 2), max(0, cy - h // 2)
                    x2, y2 = min(width, cx + w // 2), min(height, cy + h // 2)
                
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

    parser = argparse.ArgumentParser(description="Visualize Wrong-Way Driving Violations")
    # Put your video file path here:
    parser.add_argument("--video", type=str, default="data/intersection.mp4", help="Path to input video file (e.g. data/intersection.mp4)")
    # Put your trajectory tracks CSV path here:
    parser.add_argument("--tracks", type=str, default="data/tracks.csv", help="Path to trajectory tracks CSV file")
    # Put your wrong-way violations CSV path here:
    parser.add_argument("--violations", type=str, default="outputs/wrong_way.csv", help="Path to wrong-way violations CSV")
    # Put your output annotated video path here:
    parser.add_argument("--output", type=str, default="outputs/video/wrong_way_annotated.mp4", help="Path to output annotated video")
    
    args = parser.parse_args()
    visualize_wrong_way(args.video, args.tracks, args.violations, args.output)


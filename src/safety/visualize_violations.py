import argparse
import os
import cv2
import pandas as pd
import numpy as np
from collections import defaultdict

# BGR color palette matching the reference image styling
ORANGE = (0, 140, 255)  # Tailgating
YELLOW = (0, 220, 255)  # Leader (TG)
MAGENTA = (255, 0, 255) # Overtaker (OT)
CYAN = (255, 255, 0)    # Overtaken (OT)
PURPLE = (128, 0, 128)      # Sudden Braking
LIGHT_GREEN = (100, 255, 0) # Vehicle Stoppage
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

def load_tracks_df(csv_path):
    df = pd.read_csv(csv_path)
    tracks_by_frame = defaultdict(list)
    for _, row in df.iterrows():
        frame = int(row['frame'])
        tracks_by_frame[frame].append({
            'track_id': int(row['track_id']),
            'class_name': row['class_name'],
            'x1': float(row['x1']),
            'y1': float(row['y1']),
            'x2': float(row['x2']),
            'y2': float(row['y2']),
            'center_x': float(row['center_x']),
            'center_y': float(row['center_y']),
        })
    return tracks_by_frame

def load_violations(csv_path):
    df = pd.read_csv(csv_path)
    tailgating_by_frame = defaultdict(list)
    overtaking_by_frame = defaultdict(list)
    braking_by_frame = defaultdict(list)
    stoppage_by_frame = defaultdict(list)
    
    for _, row in df.iterrows():
        vtype = row['violation_type']
        f = int(row['frame']) if not pd.isna(row['frame']) else -1
        
        if vtype == 'Tailgating':
            tailgating_by_frame[f].append({
                'follower': int(row['track_id']),
                'leader': int(row['leader_track_id']),
                'd': float(row['d'])
            })
        elif vtype == 'Unsafe Overtaking':
            overtaking_by_frame[f].append({
                'overtaker': int(row['track_id']),
                'overtaken': int(row['leader_track_id']),
                'd': float(row['d'])
            })
        elif vtype == 'Sudden Braking':
            braking_by_frame[f].append({
                'track_id': int(row['track_id']),
                'd': float(row['d'])
            })
        elif vtype == 'Vehicle Stoppage':
            stoppage_by_frame[f].append({
                'track_id': int(row['track_id']),
                'd': float(row['d'])
            })
            
    return tailgating_by_frame, overtaking_by_frame, braking_by_frame, stoppage_by_frame

def main():
    parser = argparse.ArgumentParser(description="Visualize Roundabout Safe Space Violations on Video")
    parser.add_argument("--video", default=r"D:\btp\narain_data\full1 (1).MP4", help="Path to original input video")
    parser.add_argument("--tracks", default=r"D:\btp\narain_data\full1_tracks (1).csv", help="Path to tracking CSV annotations")
    parser.add_argument("--violations", default=r"D:\btp\Traffic_Object_Detection_and_Tracking\src\safety\rule.csv", help="Path to safe space violations CSV")
    parser.add_argument("--output", default=r"D:\btp\Traffic_Object_Detection_and_Tracking\outputs\video\full1_safe_space_annotated.mp4", help="Path to save annotated output video")
    parser.add_argument("--only-violations", action="store_true", help="Compile a video containing only frames with violations")
    parser.add_argument("--only-overtaking", action="store_true", help="Compile a video containing only frames with active overtaking")
    parser.add_argument("--only-braking", action="store_true", help="Compile a video containing only frames with sudden braking")
    parser.add_argument("--only-stoppage", action="store_true", help="Compile a video containing only frames with vehicle stoppage")
    parser.add_argument("--violation-type", type=str, default="All", 
                        choices=["All", "Tailgating", "Unsafe Overtaking", "Sudden Braking", "Vehicle Stoppage"],
                        help="Filter specific violation type to draw")
    
    args = parser.parse_args()
    
    print(f"[visualizer] Loading tracks from {args.tracks} ...")
    tracks_by_frame = load_tracks_df(args.tracks)
    
    print(f"[visualizer] Loading violations from {args.violations} ...")
    tailgating_by_frame, overtaking_by_frame, braking_by_frame, stoppage_by_frame = load_violations(args.violations)
    
    # Pre-calculate active overtaking frame windows (e.g. 30 frames centered on crossover)
    active_overtaking_by_frame = defaultdict(list)
    for f_cross, events in overtaking_by_frame.items():
        for ev in events:
            for f in range(max(0, f_cross - 15), f_cross + 16):
                active_overtaking_by_frame[f].append(ev)
                
    # Pre-calculate active sudden braking frame windows (e.g. 30 frames centered on braking frame)
    active_braking_by_frame = defaultdict(list)
    for f_braking, events in braking_by_frame.items():
        for ev in events:
            for f in range(max(0, f_braking - 15), f_braking + 16):
                active_braking_by_frame[f].append(ev)
                
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {args.video}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print(f"[visualizer] Processing video {args.video} ({width}x{height} @ {fps} fps) ...")
    
    frame_idx = 0
    written_count = 0
    
    while True:
        ret, img = cap.read()
        if not ret:
            break
            
        tracks = tracks_by_frame.get(frame_idx, [])
        tg_events = tailgating_by_frame.get(frame_idx, [])
        ot_events = active_overtaking_by_frame.get(frame_idx, [])
        brk_events = active_braking_by_frame.get(frame_idx, [])
        stp_events = stoppage_by_frame.get(frame_idx, [])
        
        # Build mapping of active violations per track in this frame
        violation_info = defaultdict(lambda: {
            'is_tailgating': False,
            'leader_id': None,
            'is_leader': False,
            'is_overtaker': False,
            'overtaken_id': None,
            'is_overtaken': False,
            'is_braking': False,
            'is_stoppage': False,
        })
        
        for tg in tg_events:
            fol = tg['follower']
            ld = tg['leader']
            violation_info[fol]['is_tailgating'] = True
            violation_info[fol]['leader_id'] = ld
            violation_info[ld]['is_leader'] = True
            
        for ot in ot_events:
            ovr = ot['overtaker']
            ovn = ot['overtaken']
            violation_info[ovr]['is_overtaker'] = True
            violation_info[ovr]['overtaken_id'] = ovn
            violation_info[ovn]['is_overtaken'] = True
            
        for brk in brk_events:
            tid = brk['track_id']
            violation_info[tid]['is_braking'] = True
            
        for stp in stp_events:
            tid = stp['track_id']
            violation_info[tid]['is_stoppage'] = True
            
        # Check output filtering options
        has_any_violation = (len(tg_events) > 0 or len(ot_events) > 0 or 
                             len(brk_events) > 0 or len(stp_events) > 0)
        has_active_overtaking = len(ot_events) > 0
        has_active_braking = len(brk_events) > 0
        has_active_stoppage = len(stp_events) > 0
        
        should_write = True
        if args.only_overtaking and not has_active_overtaking:
            should_write = False
        elif args.only_braking and not has_active_braking:
            should_write = False
        elif args.only_stoppage and not has_active_stoppage:
            should_write = False
        elif args.only_violations and not has_any_violation:
            should_write = False
            
        if should_write:
            # Draw connecting lines for violations first (renders underneath boxes)
            # Tailgating connection lines
            if args.violation_type in ["All", "Tailgating"]:
                for tg in tg_events:
                    fol = tg['follower']
                    ld = tg['leader']
                    t_fol = next((tk for tk in tracks if tk['track_id'] == fol), None)
                    t_ld = next((tk for tk in tracks if tk['track_id'] == ld), None)
                    if t_fol and t_ld:
                        p1 = (int(t_fol['center_x']), int(t_fol['center_y']))
                        p2 = (int(t_ld['center_x']), int(t_ld['center_y']))
                        cv2.line(img, p1, p2, ORANGE, 2, cv2.LINE_AA)
            
            # Overtaking connection lines
            if args.violation_type in ["All", "Unsafe Overtaking"]:
                for ot in ot_events:
                    ovr = ot['overtaker']
                    ovn = ot['overtaken']
                    t_ovr = next((tk for tk in tracks if tk['track_id'] == ovr), None)
                    t_ovn = next((tk for tk in tracks if tk['track_id'] == ovn), None)
                    if t_ovr and t_ovn:
                        p1 = (int(t_ovr['center_x']), int(t_ovr['center_y']))
                        p2 = (int(t_ovn['center_x']), int(t_ovn['center_y']))
                        cv2.line(img, p1, p2, MAGENTA, 2, cv2.LINE_AA)
            
            # Draw vehicle boxes and label overlays
            for t in tracks:
                tid = t['track_id']
                info = violation_info[tid]
                
                tags = []
                color = None
                priority = 0
                
                # Check for Vehicle Stoppage
                if args.violation_type in ["All", "Vehicle Stoppage"]:
                    if info['is_stoppage']:
                        tags.append("Stoppage")
                        color = LIGHT_GREEN
                        priority = 6
                        
                # Check for Sudden Braking
                if args.violation_type in ["All", "Sudden Braking"]:
                    if info['is_braking']:
                        tags.append("Braking")
                        if priority < 5:
                            color = PURPLE
                            priority = 5
                            
                # Check for Overtaking
                if args.violation_type in ["All", "Unsafe Overtaking"]:
                    if info['is_overtaker']:
                        tags.append("Overtaker (OT)")
                        if priority < 4:
                            color = MAGENTA
                            priority = 4
                    elif info['is_overtaken']:
                        tags.append("Overtaken (OT)")
                        if priority < 4:
                            color = CYAN
                            priority = 4
                

                # Check for Tailgating
                if args.violation_type in ["All", "Tailgating"]:
                    if info['is_tailgating']:
                        tags.append("Tailgating")
                        if priority < 2:
                            color = ORANGE
                            priority = 2
                    elif info['is_leader']:
                        tags.append("Leader (TG)")
                        if priority < 1:
                            color = YELLOW
                            priority = 1
                            
                if not tags:
                    continue  # Only draw vehicles that have active violations matching filter
                    
                x1, y1, x2, y2 = int(t['x1']), int(t['y1']), int(t['x2']), int(t['y2'])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                label_text = f"ID:{tid} " + " | ".join(tags)
                (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                
                ty = y1 - 5
                if ty - th - 4 < 0:
                    ty = y1 + th + 5
                    
                cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + baseline + 2), color, -1)
                cv2.putText(img, label_text, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLACK, 1, cv2.LINE_AA)
                
            # Draw HUD Frame indicator
            cv2.putText(img, f"Frame: {frame_idx}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
            out.write(img)
            written_count += 1
            
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"[visualizer] processed {frame_idx}/{total_frames} frames ...")
            
    cap.release()
    out.release()
    print(f"[visualizer] Successfully saved annotated video -> {args.output} ({written_count} frames written)")

if __name__ == "__main__":
    main()

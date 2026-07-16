import pandas as pd
import numpy as np

def detect_wrong_way_violations(csv_file, output_csv=None):
    # Load dataset
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        return

    try:
        from .calibration import CENTER_X as X_c, CENTER_Y as Y_c, R_INNER as R_MIN, R_OUTER as R_MAX
    except ImportError:
        from calibration import CENTER_X as X_c, CENTER_Y as Y_c, R_INNER as R_MIN, R_OUTER as R_MAX

    FPS = 30.0
    OMEGA_THRESHOLD = -0.1
    CONSECUTIVE_FRAMES_THRESHOLD = 15

    # 1. Calculate Polar Radius 'r' and Angle 'theta'
    # Assuming the coordinates to use are world_x and world_y
    df['dx'] = df['world_x'] - X_c
    df['dy'] = df['world_y'] - Y_c
    df['r'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['theta'] = np.arctan2(df['dy'], df['dx'])

    # 2. Group by 'track_id' and sort by 'frame'
    df = df.sort_values(by=['track_id', 'frame'])

    # Flag to keep track of violations
    violations = {}

    for track_id, group in df.groupby('track_id'):
        group = group.copy()
        
        # 3. Compute shortest angular change
        theta_shift = group['theta'].shift(1)
        group['delta_theta'] = np.arctan2(np.sin(group['theta'] - theta_shift), np.cos(group['theta'] - theta_shift))
        
        # 4. Calculate angular velocity
        group['omega'] = group['delta_theta'] * FPS
        
        # 5. Check if vehicle is inside the circulating ring and traveling wrong way
        group['is_in_ring'] = (group['r'] >= R_MIN) & (group['r'] <= R_MAX)
        group['is_wrong_way'] = (group['omega'] < OMEGA_THRESHOLD) & group['is_in_ring']
        
        # 6. Track continuous frames
        # We can find consecutive True values in 'is_wrong_way' by grouping by consecutive blocks
        group['consecutive_group'] = (group['is_wrong_way'] != group['is_wrong_way'].shift()).cumsum()
        
        # Filter only the wrong way frames
        wrong_way_frames = group[group['is_wrong_way']]
        
        # Count consecutive frames
        if not wrong_way_frames.empty:
            counts = wrong_way_frames.groupby('consecutive_group').size()
            valid_groups = counts[counts >= CONSECUTIVE_FRAMES_THRESHOLD].index
            
            if not valid_groups.empty:
                # Find the first frame of the first valid sequence
                first_valid_group = valid_groups[0]
                violation_start_frame = wrong_way_frames[wrong_way_frames['consecutive_group'] == first_valid_group]['frame'].iloc[0]
                class_name = group['class_name'].iloc[0]
                
                violations[track_id] = {
                    'class_name': class_name,
                    'start_frame': int(violation_start_frame)
                }

    # Prepare outputs
    if not violations:
        print("No Wrong-Way Driving Violations detected.")
        return

    print("--- Wrong-Way Driving Violations Summary ---")
    
    # Broken down by class_name
    class_counts = {}
    for vid, info in violations.items():
        c_name = info['class_name']
        class_counts[c_name] = class_counts.get(c_name, 0) + 1
        
    print("\nTotal number of unique track IDs flagged by class:")
    for c_name, count in class_counts.items():
        print(f"- {c_name}: {count}")

    print("\nSpecific track IDs caught violating the rule:")
    for track_id, info in violations.items():
        print(f"Track ID {track_id} (Class: {info['class_name']}) - Violation started at frame: {info['start_frame']}")

    if output_csv:
        records = []
        for track_id, info in violations.items():
            records.append({
                'track_id': track_id,
                'class_name': info['class_name'],
                'start_frame': info['start_frame'],
                'violation_type': 'Wrong-Way'
            })
        pd.DataFrame(records).to_csv(output_csv, index=False)
        print(f"\nSaved wrong-way violations to {output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Wrong-way violation detector")
    parser.add_argument("--csv", type=str, default=r"D:\btp\narain_data\full1_tracks (1).csv")
    parser.add_argument("--output", type=str, default=r"D:\btp\Traffic_Object_Detection_and_Tracking\src\safety\wrong_way.csv")
    args = parser.parse_args()
    
    detect_wrong_way_violations(args.csv, args.output)

import argparse
import pandas as pd
import numpy as np
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description="Evaluate Vehicle Stoppage Rule")
parser.add_argument("--csv", type=str, default=None, help="Path to input tracks CSV")
parser.add_argument("--output", type=str, default=os.path.join(os.path.dirname(__file__), 'csv_outputs', 'stoppage_violations.csv'), help="Path to output stoppage violations CSV")
args, unknown = parser.parse_known_args()

csv_path = args.csv
output_path = args.output

if csv_path is None:
    csv_path = r'D:\btp\narain_data\long1_tracks_narain_cleaned_edited.csv'
    if not os.path.exists(csv_path):
        # Try looking in the workspace relative to this file
        possible_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'csv', 'long1_tracks_narain_cleaned_edited.csv')
        if os.path.exists(possible_path):
            csv_path = possible_path
        else:
            # Try local path from project root
            possible_path = os.path.join('src', 'outputs', 'csv', 'long1_tracks_narain_cleaned_edited.csv')
            if os.path.exists(possible_path):
                csv_path = possible_path

# Fallback check
if not csv_path or not os.path.exists(csv_path):
    csv_path = r'D:\btp\narain_data\long1_tracks_narain_cleaned_edited.csv'

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: The file '{csv_path}' was not found.")
    exit(1)

try:
    from .calibration import CENTER_X, CENTER_Y, R_INNER, R_OUTER, R_SEPARATOR
except ImportError:
    from calibration import CENTER_X, CENTER_Y, R_INNER, R_OUTER, R_SEPARATOR

X_c = CENTER_X
Y_c = CENTER_Y

# Step 1: Coordinate Conversion & Lane Assignment
df['r'] = np.sqrt((df['world_x'] - X_c)**2 + (df['world_y'] - Y_c)**2)

def assign_lane(r):
    if R_INNER <= r < R_SEPARATOR:
        return 'Inner'
    elif R_SEPARATOR <= r <= R_OUTER:
        return 'Outer'
    else:
        return 'None'

df['lane'] = df['r'].apply(assign_lane)

# Filter out vehicles not in the circulating ring
ring_df = df[df['lane'] != 'None'].copy()

# Sort by track_id and frame
ring_df = ring_df.sort_values(by=['track_id', 'frame'])

stoppage_records = []

# Group by track_id
for track_id, group in ring_df.groupby('track_id'):
    group = group.copy()
    if len(group) < 90:
        continue
        
    prev_x = group['world_x'].shift(90)
    prev_y = group['world_y'].shift(90)
    disp = np.hypot(group['world_x'] - prev_x, group['world_y'] - prev_y)
    window_mean_vel = group['velocity_ms'].rolling(90).mean()
    
    stoppage_mask = (disp < 1.0) & (window_mean_vel < 0.8)
    
    violating_rows = group[stoppage_mask]
    for _, row in violating_rows.iterrows():
        # Get matching displacement value
        d_val = disp.loc[row.name]
        stoppage_records.append({
            'violation_type': 'Vehicle Stoppage',
            'frame': int(row['frame']),
            'track_id': int(track_id),
            'leader_track_id': None,
            'class_name': row['class_name'],
            'lane': row['lane'],
            'd': d_val
        })

stoppage_df = pd.DataFrame(stoppage_records)

# Summary outputs
print(f"Total unique track IDs in circulating ring: {ring_df['track_id'].nunique()}")
print(f"Detected {len(stoppage_df)} Vehicle Stoppage violations.")

if not stoppage_df.empty:
    unique_stoppage = stoppage_df.drop_duplicates('track_id')
    print("\nStoppage Violations by class_name:")
    print(unique_stoppage['class_name'].value_counts().to_string())
else:
    print("No stoppage violations found.")

# Export to output_path
os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
if not stoppage_df.empty:
    columns_order = ['violation_type', 'frame', 'track_id', 'leader_track_id', 'class_name', 'lane', 'd']
    stoppage_df = stoppage_df[columns_order]
stoppage_df.to_csv(output_path, index=False)
print(f"\nStoppage violations have been successfully saved to '{output_path}'.")

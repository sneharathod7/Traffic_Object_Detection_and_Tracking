import argparse
import pandas as pd
import numpy as np
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description="Evaluate Tailgating Safe Space Rule")
parser.add_argument("--csv", type=str, default=None, help="Path to input tracks CSV")
parser.add_argument("--output", type=str, default=os.path.join(os.path.dirname(__file__), 'csv_outputs', 'tailgating_violations.csv'), help="Path to output tailgating violations CSV")
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

# If path still doesn't exist, fallback to D:\btp\narain_data\long1_tracks_narain_cleaned_edited.csv
if not csv_path or not os.path.exists(csv_path):
    for possible_fallback in [r'D:\btp\narain_data\long1_tracks_narain_cleaned_edited.csv']:
        if os.path.exists(possible_fallback):
            csv_path = possible_fallback
            break

df = pd.read_csv(csv_path)

try:
    from .calibration import CENTER_X, CENTER_Y, R_INNER, R_OUTER, R_SEPARATOR
except ImportError:
    from calibration import CENTER_X, CENTER_Y, R_INNER, R_OUTER, R_SEPARATOR

X_c = CENTER_X
Y_c = CENTER_Y
fps = 30
dt = 1/30

# Step 1: Coordinate Conversion & Lane Assignment
df['r'] = np.sqrt((df['world_x'] - X_c)**2 + (df['world_y'] - Y_c)**2)
df['theta'] = np.arctan2(df['world_y'] - Y_c, df['world_x'] - X_c)

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

# 1. Total unique track IDs that entered the circulating ring
unique_ring_tracks = ring_df['track_id'].nunique()

# Step 2: Detect Tailgating / Proximity Violation
tailgating_records = []

# Group frame-by-frame, then by lane
for (frame, lane), group in ring_df.groupby(['frame', 'lane']):
    if len(group) < 2:
        continue
    # Sort by theta
    sorted_group = group.sort_values('theta').to_dict('records')
    n = len(sorted_group)
    
    for i in range(n):
        follower = sorted_group[i]
        leader = sorted_group[(i + 1) % n]
        
        delta_theta = (leader['theta'] - follower['theta']) % (2 * np.pi)
        d = ((leader['r'] + follower['r']) / 2) * delta_theta
        
        if d < 4.0 and follower['velocity_ms'] > 1.0:
            tailgating_records.append({
                'frame': frame,
                'follower_track_id': follower['track_id'],
                'leader_track_id': leader['track_id'],
                'lane': lane,
                'd': d,
                'class_name': follower['class_name']
            })

tailgating_df = pd.DataFrame(tailgating_records)

# Outputs
print(f"1. Total unique track IDs that entered the circulating ring: {unique_ring_tracks}")

print("\n2. Tailgating Violations by class_name:")
if not tailgating_df.empty:
    unique_tailgating = tailgating_df.drop_duplicates('follower_track_id')
    print(unique_tailgating['class_name'].value_counts().to_string())
else:
    print("None")

print("\n3. Sample summary dataframe showing 10 random frames where a Tailgating Violation occurred:")
if not tailgating_df.empty:
    sample_size = min(10, len(tailgating_df))
    sample_df = tailgating_df[['frame', 'follower_track_id', 'leader_track_id', 'lane', 'd']].sample(sample_size, random_state=42)
    print(sample_df.to_string(index=False))
else:
    print("No tailgating violations found.")

# Exporting violations to output_path
if not tailgating_df.empty:
    tailgating_export = tailgating_df.copy()
    tailgating_export['violation_type'] = 'Tailgating'
    tailgating_export = tailgating_export.rename(columns={'follower_track_id': 'track_id'})
else:
    tailgating_export = pd.DataFrame()

combined_df = tailgating_export

if not combined_df.empty:
    columns_order = ['violation_type', 'frame', 'track_id', 'leader_track_id', 'class_name', 'lane', 'd']
    # Only keep columns that exist
    columns_order = [c for c in columns_order if c in combined_df.columns]
    combined_df = combined_df[columns_order]

os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
combined_df.to_csv(output_path, index=False)
print(f"\n4. Tailgating violations have been successfully saved to '{output_path}'.")

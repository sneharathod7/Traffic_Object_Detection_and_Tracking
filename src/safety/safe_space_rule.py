import os
import pandas as pd
import numpy as np

def _resolve(path_str: str) -> str:
    if not path_str or os.path.exists(path_str):
        return path_str
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    cands = [
        os.path.join(project_root, path_str),
        os.path.join(project_root, "newsafety_rules", path_str),
        os.path.join(project_root, "newsafety_rules", "data", os.path.basename(path_str)),
        os.path.join(project_root, "data", os.path.basename(path_str)),
        os.path.join(script_dir, "..", path_str),
    ]
    for c in cands:
        if os.path.exists(c):
            return os.path.abspath(c)
    return path_str

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Tailgating / Safe-Space Rule Violation Detection")
    # Put your input trajectory CSV path here:
    parser.add_argument("--tracks", type=str, default="data/tracks.csv", help="Path to input trajectory CSV file (e.g. data/tracks.csv)")
    # Put your output violations CSV path here:
    parser.add_argument("--output", type=str, default="outputs/tailgating_violations.csv", help="Path to output violations CSV file")
    return parser.parse_args()

args = parse_args()
# Put the path to your input trajectory CSV file here:
csv_path = args.tracks
resolved_csv = _resolve(csv_path)

if not os.path.exists(resolved_csv):
    # Fallback to local data candidates if default doesn't exist
    fallback_cands = [
        "data/long1_tracks_narain_cleaned_edited.csv",
        "newsafety_rules/data/long1_tracks_narain_cleaned_edited.csv",
    ]
    for cand in fallback_cands:
        if os.path.exists(_resolve(cand)):
            resolved_csv = _resolve(cand)
            break

df = pd.read_csv(resolved_csv)

# Parameters
X_c = 43.5
Y_c = 28.5
fps = 30
dt = 1/30

# Step 1: Coordinate Conversion & Lane Assignment
df['r'] = np.sqrt((df['world_x'] - X_c)**2 + (df['world_y'] - Y_c)**2)
df['theta'] = np.arctan2(df['world_y'] - Y_c, df['world_x'] - X_c)

def assign_lane(r):
    if 6.0 <= r < 10.0:
        return 'Inner'
    elif 10.0 <= r <= 14.0:
        return 'Outer'
    else:
        return 'None'

df['lane'] = df['r'].apply(assign_lane)

# Filter out vehicles not in the circulating ring
ring_df = df[df['lane'] != 'None'].copy()

# 1. Total unique track IDs that entered the circulating ring
unique_ring_tracks = ring_df['track_id'].nunique()

# Step 2: Detect Part B - Tailgating / Proximity Violation
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
    # We want unique track IDs flagged for tailgating
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

# Export to rule.csv
if not tailgating_df.empty:
    tailgating_export = tailgating_df.copy()
    tailgating_export['violation_type'] = 'Tailgating'
    tailgating_export = tailgating_export.rename(columns={'follower_track_id': 'track_id'})
    columns_order = ['violation_type', 'frame', 'track_id', 'leader_track_id', 'class_name', 'lane', 'd']
    combined_export = tailgating_export[columns_order]
else:
    combined_export = pd.DataFrame(columns=['violation_type', 'frame', 'track_id', 'leader_track_id', 'class_name', 'lane', 'd'])

combined_export.to_csv(args.output, index=False)
print(f"\n4. All tailgating violations have been successfully saved to '{args.output}'.")

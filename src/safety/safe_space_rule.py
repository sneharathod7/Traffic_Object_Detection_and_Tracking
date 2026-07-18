import argparse
import pandas as pd
import numpy as np
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description="Evaluate Safe Space Rules")
parser.add_argument("--csv", type=str, default=None, help="Path to input tracks CSV")
parser.add_argument("--output", type=str, default="rule.csv", help="Path to output violations CSV")
args, unknown = parser.parse_known_args()

csv_path = args.csv
output_path = args.output

if csv_path is None:
    csv_path = r'C:\Users\sneha\Downloads\test1_slowed_tracks (1).csv'
    if not os.path.exists(csv_path):
        # Try looking in the workspace relative to this file
        possible_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'csv', 'test1_slowed_tracks (1).csv')
        if os.path.exists(possible_path):
            csv_path = possible_path
        else:
            # Try local path from project root
            possible_path = os.path.join('src', 'outputs', 'csv', 'test1_slowed_tracks (1).csv')
            if os.path.exists(possible_path):
                csv_path = possible_path

# If path still doesn't exist, fallback to D:\btp\narain_data\full1_tracks (1).csv or test1.csv
if not csv_path or not os.path.exists(csv_path):
    for possible_fallback in [r'D:\btp\narain_data\full1_tracks (1).csv', r'D:\btp\narain_data\test1.csv']:
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


# Step 3: Detect Part B - Tailgating / Proximity Violation
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

# Step 4: Detect Part C - Unsafe Overtaking Violation
overtaking_records = []
pair_states = {} # (track_a, track_b) -> (prev_frame, prev_lane, prev_diff)

# Ensure sorting by frame for sequential tracking
ring_df = ring_df.sort_values('frame')

for frame, group in ring_df.groupby('frame'):
    for lane in ['Inner', 'Outer']:
        lane_vehicles = group[group['lane'] == lane].to_dict('records')
        n = len(lane_vehicles)
        if n < 2:
            continue
        
        for i in range(n):
            for j in range(i + 1, n):
                v1 = lane_vehicles[i]
                v2 = lane_vehicles[j]
                
                track_a = min(v1['track_id'], v2['track_id'])
                track_b = max(v1['track_id'], v2['track_id'])
                
                if v1['track_id'] == track_a:
                    rec_a = v1
                    rec_b = v2
                else:
                    rec_a = v2
                    rec_b = v1
                
                theta_a = rec_a['theta']
                theta_b = rec_b['theta']
                
                # Relative angular difference (shortest angular distance)
                diff = np.arctan2(np.sin(theta_b - theta_a), np.cos(theta_b - theta_a))
                
                pair_key = (track_a, track_b)
                if pair_key in pair_states:
                    prev_frame, prev_lane, prev_diff = pair_states[pair_key]
                    
                    if prev_frame == frame - 1 and prev_lane == lane:
                        # Check sign change
                        if prev_diff * diff < 0:
                            # 1. Filter out boundary wrap-arounds at +/- pi (crossovers must happen near 0 angle diff)
                            ang_change = np.abs(np.arctan2(np.sin(diff - prev_diff), np.cos(diff - prev_diff)))
                            # 2. Filter out vehicles that are not physically close (within 10.0 meters)
                            dist = np.hypot(rec_a['world_x'] - rec_b['world_x'], rec_a['world_y'] - rec_b['world_y'])
                            
                            if ang_change < 1.0 and dist < 10.0:
                                # If prev_diff > 0: track_b was ahead of track_a. Now diff < 0: track_b is behind track_a.
                                # So track_a overtook track_b.
                                if prev_diff > 0:
                                    overtaker = rec_a
                                    overtaken = rec_b
                                else:
                                    overtaker = rec_b
                                    overtaken = rec_a
                                
                                # Only log if the overtaker is actually moving
                                if overtaker['velocity_ms'] > 1.0:
                                    overtaking_records.append({
                                        'frame': frame,
                                        'track_id': overtaker['track_id'],
                                        'leader_track_id': overtaken['track_id'],
                                        'lane': lane,
                                        'd': np.abs(rec_a['r'] - rec_b['r']), # radial distance difference at crossover
                                        'class_name': overtaker['class_name']
                                    })
                
                pair_states[pair_key] = (frame, lane, diff)

overtaking_df = pd.DataFrame(overtaking_records)

# Step 5: Detect Part D - Sudden Braking Violation
braking_records = []
for track_id, group in ring_df.groupby('track_id'):
    group = group.sort_values('frame')
    if len(group) < 2:
        continue
    
    # Compute 7-frame rolling average velocity to smooth tracking jitter
    smooth_vel = group['velocity_ms'].rolling(window=7, min_periods=1).mean()
    # Compute acceleration (m/s^2)
    accel = smooth_vel.diff() * fps
    prev_smooth_vel = smooth_vel.shift(1)
    
    # Deceleration threshold of -6.0 m/s^2, initial speed threshold of 3.0 m/s
    braking_mask = (accel < -6.0) & (prev_smooth_vel > 3.0)
    for idx in group[braking_mask].index:
        row = group.loc[idx]
        braking_records.append({
            'frame': int(row['frame']),
            'track_id': int(track_id),
            'leader_track_id': np.nan,
            'lane': row['lane'],
            'd': float(accel.loc[idx]), # Save acceleration value in 'd'
            'class_name': row['class_name']
        })
braking_df = pd.DataFrame(braking_records)

# Step 6: Detect Part E - Vehicle Stoppage Violation
stoppage_records = []
for track_id, group in ring_df.groupby('track_id'):
    group = group.sort_values('frame')
    if len(group) < 90:
        continue
    
    # Check 90-frame (3-second) spatial displacement
    prev_x = group['world_x'].shift(90)
    prev_y = group['world_y'].shift(90)
    disp = np.hypot(group['world_x'] - prev_x, group['world_y'] - prev_y)
    window_mean_vel = group['velocity_ms'].rolling(90).mean()
    
    # If displacement is < 1.0 meter and mean speed is < 0.8 m/s
    stoppage_mask = (disp < 1.0) & (window_mean_vel < 0.8)
    for idx in group[stoppage_mask].index:
        row = group.loc[idx]
        stoppage_records.append({
            'frame': int(row['frame']),
            'track_id': int(track_id),
            'leader_track_id': np.nan,
            'lane': row['lane'],
            'd': float(disp.loc[idx]), # Save displacement in 'd'
            'class_name': row['class_name']
        })
stoppage_df = pd.DataFrame(stoppage_records)

# Outputs
print(f"1. Total unique track IDs that entered the circulating ring: {unique_ring_tracks}")



print("\n3. Tailgating Violations by class_name:")
if not tailgating_df.empty:
    unique_tailgating = tailgating_df.drop_duplicates('follower_track_id')
    print(unique_tailgating['class_name'].value_counts().to_string())
else:
    print("None")

print("\n4. Unsafe Overtaking Violations by class_name:")
if not overtaking_df.empty:
    unique_overtaking = overtaking_df.drop_duplicates('track_id')
    print(unique_overtaking['class_name'].value_counts().to_string())
else:
    print("None")

print("\n5. Sudden Braking Violations by class_name:")
if not braking_df.empty:
    unique_braking = braking_df.drop_duplicates('track_id')
    print(unique_braking['class_name'].value_counts().to_string())
else:
    print("None")

print("\n6. Vehicle Stoppage Violations by class_name:")
if not stoppage_df.empty:
    unique_stoppage = stoppage_df.drop_duplicates('track_id')
    print(unique_stoppage['class_name'].value_counts().to_string())
else:
    print("None")

print("\n7. Sample summary dataframe showing 10 random frames where a Tailgating Violation occurred:")
if not tailgating_df.empty:
    sample_size = min(10, len(tailgating_df))
    sample_df = tailgating_df[['frame', 'follower_track_id', 'leader_track_id', 'lane', 'd']].sample(sample_size, random_state=42)
    print(sample_df.to_string(index=False))
else:
    print("No tailgating violations found.")

print("\n8. Sample summary dataframe showing up to 10 frames where an Unsafe Overtaking Violation occurred:")
if not overtaking_df.empty:
    sample_size = min(10, len(overtaking_df))
    sample_df = overtaking_df[['frame', 'track_id', 'leader_track_id', 'lane', 'd']].sample(sample_size, random_state=42)
    print(sample_df.to_string(index=False))
else:
    print("No unsafe overtaking violations found.")

print("\n9. Sample summary dataframe showing up to 10 frames where a Sudden Braking Violation occurred:")
if not braking_df.empty:
    sample_size = min(10, len(braking_df))
    sample_df = braking_df[['frame', 'track_id', 'lane', 'd']].sample(sample_size, random_state=42)
    print(sample_df.to_string(index=False))
else:
    print("No sudden braking violations found.")

print("\n10. Sample summary dataframe showing up to 10 frames where a Vehicle Stoppage Violation occurred:")
if not stoppage_df.empty:
    sample_size = min(10, len(stoppage_df))
    sample_df = stoppage_df[['frame', 'track_id', 'lane', 'd']].sample(sample_size, random_state=42)
    print(sample_df.to_string(index=False))
else:
    print("No vehicle stoppage violations found.")


if not tailgating_df.empty:
    tailgating_export = tailgating_df.copy()
    tailgating_export['violation_type'] = 'Tailgating'
    tailgating_export = tailgating_export.rename(columns={'follower_track_id': 'track_id'})
else:
    tailgating_export = pd.DataFrame()

if not overtaking_df.empty:
    overtaking_export = overtaking_df.copy()
    overtaking_export['violation_type'] = 'Unsafe Overtaking'
else:
    overtaking_export = pd.DataFrame()

if not braking_df.empty:
    braking_export = braking_df.copy()
    braking_export['violation_type'] = 'Sudden Braking'
else:
    braking_export = pd.DataFrame()

if not stoppage_df.empty:
    stoppage_export = stoppage_df.copy()
    stoppage_export['violation_type'] = 'Vehicle Stoppage'
else:
    stoppage_export = pd.DataFrame()

combined_df = pd.concat([tailgating_export, overtaking_export, braking_export, stoppage_export], ignore_index=True)
# Reorder columns for better readability
columns_order = ['violation_type', 'frame', 'track_id', 'leader_track_id', 'class_name', 'lane', 'd']
# Only keep columns that exist
columns_order = [c for c in columns_order if c in combined_df.columns]
combined_df = combined_df[columns_order]

combined_df.to_csv(output_path, index=False)
print(f"\n11. All violations have been successfully saved to '{output_path}'.")

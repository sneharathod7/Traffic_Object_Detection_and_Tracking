# Pipeline Step 2: Tracking (10-Layer Architecture)

## Architectural Overview
The tracker module (`src/tracker.py`) orchestrates the temporal association of detections into persistent vehicle trajectories. We implement a custom **BoT-SORT / StrongSORT hybrid** wrapped inside a classic ByteTrack 2-stage matching loop.

The tracker models the state of each object using an 8-dimensional state vector in a discrete-time linear **Kalman Filter**:
`State Vector: X = [cx, cy, w, h, vcx, vcy, vw, vh]`
Where `(cx, cy)` is the center of the bounding box, `(w, h)` is the box size, and the `v`-prefixed variables represent the velocities.

Because Indian drone traffic footage suffers from immense occlusion density and bounding box collision, a standard intersection-over-union (IoU) tracker immediately fails. We engineered **10 Layers of Sophistication** to enforce strict physical and visual constraints.

---

## 1. Distance-IoU (DIoU) Assignment
Standard IoU fails when a small vehicle (motorcycle) is enclosed by a larger vehicle's (bus) bounding box.
In `src/tracker.py -> diou_matrix`, we calculate DIoU as:
```math
DIoU = IoU - \frac{d^2}{c^2}
```
Where:
- `d` = Euclidean distance between the center points of the two bounding boxes.
- `c` = Diagonal length of the smallest enclosing box covering both bounding boxes.

This actively penalizes assignments where the bounding box centers don't align, physically blocking collision-based ID swaps. The DIoU metric ranges from `[-1, 1]`, and is inverted into a cost matrix ranging from `[0, 2]`.

## 2. Tri-Modal Appearance Extraction (Global ReID)
Implemented in `src/reid.py -> ReIDExtractor`.
For every detection where `confidence >= low_thresh (0.10)`, we crop the bounding box and compute a **Tri-Modal Feature Embedding**:
1. **Spatial Features**: Extracted via a `ResNet50` backbone, pooling into a 128-dimensional L2-normalized embedding.
2. **Color Features**: A 3D HSV Color Histogram (8 Hue × 8 Saturation × 8 Value bins = 512 dimensions), normalized using `cv2.normalize`.
3. **Temporal Features (EMA)**: Handled by `STrack.update_appearance()`. We apply an Exponential Moving Average to slowly blend new observations into the historical embedding to handle smooth lighting transitions:
   `curr_emb = 0.85 * curr_emb + 0.15 * new_emb`

## 3. 5-Component Cost Matrix Fusion
During Stage 1 matching (`tracker.py -> hungarian_match`), the cost matrix is an explicit mathematical fusion. 
First, we apply a dynamic motion gate `G`. If `det_cls_id == 3` (motorcycle), `G = 4.5 * max(w, h)`, else `G = 2.5 * max(w, h)`. If `d_center > G`, the match is rejected (`cost = 1.0`).

If it passes the gate, the total assignment cost is computed as:
```python
# cost_motion ranges [0, 1] based on proximity to the motion gate G
cost_motion = d_center / G

# d_app is a weighted combination of Cosine Distances for ResNet50 and HSV Histograms
d_app = compute_appearance_distance(det["emb"], trk.curr_emb)

# The Fused Cost Formula:
cost[r, c] = 0.4 * (1.0 - diou_mat[r, c]) + 0.2 * cost_motion + 0.4 * d_app
```
This forces the Hungarian algorithm (via `scipy.optimize.linear_sum_assignment`) to respect visual identity just as strictly as spatial bounding box overlap.

## 4. Class-Specific Kalman Physics
Implemented in `STrack._update_noise_matrices()`. 
Standard Kalman filters assume all objects share identical motion characteristics. We inject class-specific structural knowledge into the Process Noise Matrix (`Q`).
- **Base Process Noise (`p_std, v_std`)**: Derived from standard SORT scaling.
- **Motorcycles (`class_id == 3`)**: Extremely agile. We multiply position variance by `1.5` and velocity variance by `2.0`.
- **Buses (`class_id == 5`)**: Heavy momentum. We multiply position variance by `0.5` and velocity variance by `0.2` (preventing the Kalman filter from predicting impossible sharp turns).

## 5. Adaptive Uncertainty (Confidence Scaling)
Implemented in `STrack._update_noise_matrices()`.
When an object passes through a shadow, YOLO's bounding box output becomes jittery and its confidence score plummets. 
We exponentially scale the Kalman Measurement Noise Matrix (`R`) based on YOLO's confidence:
```python
uncertainty_factor = 1.0 + (1.0 - confidence) * 5.0
R = np.diag([p_std, p_std, p_std, p_std]) ** 2 * uncertainty_factor
```
A 0.20 confidence box scales the measurement uncertainty by a factor of `5.0`. The Kalman filter optimally responds by discarding the visual measurement and coasting on the velocity vector.

## 6. Adaptive Track Buffering
Implemented in `STrack.get_max_lost_frames()`.
Instead of a static 30-frame track buffer, we compute `max_lost_frames` dynamically:
- Short/unreliable tracks (`tracklet_len < 30`) or low-confidence tracks decay the buffer down to `0.5x`.
- Long-established tracks (`tracklet_len >= 30`) scale the buffer up to `2.0x`.
- Long-tracked motorcycles are given an absolute minimum `1.5x` buffer bonus to handle complete occlusions behind buses.

## 7. OC-SORT Observation-Centric Online Smoothing (OOS)
Implemented in `STrack.re_activate()`.
When a track is lost for a 15-frame occlusion gap, the Kalman velocity vector drifts wildly, causing an immediate failure upon reconnection.
To fix this, we apply an Observation-Centric state repair. Upon matching a detection after a gap `> 1` frame, we *discard* the drifted Kalman prediction, and manually compute the true virtual velocity:
```python
vx = (new_cx - last_valid_cx) / gap
vy = (new_cy - last_valid_cy) / gap
self.kf.statePost[0, 0] = new_cx
self.kf.statePost[4, 0] = vx
self.kf.errorCovPost = np.eye(8) * 10.0 # Reset covariance to trust the repair
```

## 8. OC-SORT Observation-Centric Recovery (OCR)
Implemented in `BYTETracker.update() -> Stage 3`.
Standard trackers search for lost tracks around their drifted Kalman predictions. This leads to 0% recovery rates for non-linear motion.
OCR completely ignores the Kalman prediction. It defines a dynamic spatial search radius exclusively around `STrack.last_valid_cx` (the last known true visual observation).
```python
max_dist = max(3.0 * sz, 1.5 * sz * np.sqrt(curr_gap))
d_spatial = np.hypot(det_cx - t.last_valid_cx, det_cy - t.last_valid_cy)
```

## 9. Active Track Deduplication (IoU-NMS Ghost Suppression)
Implemented in `BYTETracker._suppress_duplicate_tracks(iou_threshold=0.65)`.
SAHI slicing creates physical artifacts where one large vehicle generates two valid detections on the stitching boundary. If both survive `sahi_fusion.py`, they generate two separate tracks for the same vehicle.
At the very end of the tracker loop, we run an `iou_matrix` pairwise comparison across all **active** tracks.
If `IoU(T1, T2) > 0.65`, it is physically impossible for them to be two different rigid vehicles. 
The track with the shorter `tracklet_len` is marked `TrackState.Removed` and instantly evicted, preventing count pollution.

## 10. Nascent ID Switch Repair
Implemented in `BYTETracker._repair_nascent_id_switches(nascent_window=5, max_recent_loss=3)`.
Detection jitter (e.g. motion blur) can cause an object to fail the DIoU threshold for exactly 1 frame. The track goes `Lost`, and on the next frame, the object initializes a brand new track ID (an ID switch).
We scan all new tracks (`tracklet_len <= 5`) against recently lost tracks (`lost_frames <= 3`).
If we find a pair that satisfies:
1. Exact same `class_id`
2. `d_spatial <= 2.0 * max(w, h)`
3. `d_app < 0.30`
We retroactively perform a memory swap. The new track absorbs the old track's `track_id`, historical `tracklet_len`, and EMA appearance embedding. The old track is instantly removed from `lost_stracks`. The fragmented ID is repaired on-the-fly.

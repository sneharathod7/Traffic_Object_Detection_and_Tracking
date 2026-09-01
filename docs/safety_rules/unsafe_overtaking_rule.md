# Unsafe Overtaking Detection Rule

This document details the algorithm and implementation of the **Unsafe Overtaking Detection Rule** within the restricted roundabout zone.

---

## 1. Physical Context & Rule Description
Roundabouts are high-conflict areas where lane changing and overtaking should be restricted to prevent sideswipe and T-bone collisions. 

The rule states:
> A vehicle is flagged for an unsafe overtaking violation if it approaches another vehicle from behind traveling in a similar direction, performs a lane-change or lateral shift, and successfully passes the vehicle within the restricted roundabout zone ($r < 14.0$ meters).

---

## 2. Mathematical Formulation

### A. Heading Alignment
We compute the global direction vector for each vehicle track $A$ using its start and end coordinates:
$$\vec{d}_A = (x_{\text{end}} - x_{\text{start}}, y_{\text{end}} - y_{\text{start}})$$
$$\hat{d}_A = \frac{\vec{d}_A}{\|\vec{d}_A\|}$$

We only evaluate vehicle pairs $A$ and $B$ if they are traveling in approximately the same direction:
$$\theta_{\text{heading}} = \arccos(\hat{d}_A \cdot \hat{d}_B) < 30^\circ$$

### B. Spatial Projections (Forward & Lateral)
For frames where both vehicles are active, we compute the relative position vector:
$$\vec{rel} = \vec{pos}_A - \vec{pos}_B$$

We define the average movement vector of the pair as:
$$\vec{dir}_{\text{rel}} = \frac{\hat{d}_A + \hat{d}_B}{2}$$
$$\hat{dir}_{\text{rel}} = \frac{\vec{dir}_{\text{rel}}}{\|\vec{dir}_{\text{rel}}\|}$$

We project the relative position onto this reference frame:
1. **Forward Distance ($d_{\text{forward}}$)**:
   $$d_{\text{forward}} = \vec{rel} \cdot \hat{dir}_{\text{rel}}$$
   * If $d_{\text{forward}} < -0.5$ meters, Vehicle $A$ is **behind** Vehicle $B$.
   * If $d_{\text{forward}} > 0.5$ meters, Vehicle $A$ is **ahead of** Vehicle $B$.
   * If $|d_{\text{forward}}| \le 0.5$ meters, the vehicles are **parallel/crossing**.

2. **Lateral Offset ($d_{\text{lateral}}$)**:
   Using the 2D cross product:
   $$d_{\text{lateral}} = \hat{dir}_{\text{rel}} \times \vec{rel} = \hat{dir}_{\text{rel},x} \cdot \vec{rel}_y - \hat{dir}_{\text{rel},y} \cdot \vec{rel}_x$$

### C. Overtaking Sequence Validation
An overtaking maneuver is valid if the following conditions are met in sequence over time:
1. **Initial State**: Vehicle $A$ must start behind Vehicle $B$ ($d_{\text{forward}} < -0.5$).
2. **Final State**: Vehicle $A$ must end up ahead of Vehicle $B$ ($d_{\text{forward}} > 0.5$).
3. **Passing Frame**: There must be a frame where the vehicles cross ($|d_{\text{forward}}| \le 0.5$).
4. **Distance Convergence/Divergence**: The physical distance between the vehicles must decrease as $A$ approaches $B$, and then increase after passing.
5. **Lateral Displacement**: The lateral offset must change during the maneuver, signifying a steering path change:
   $$\Delta d_{\text{lateral}} = \left| \text{median}(|d_{\text{lateral, post}}|) - \text{median}(|d_{\text{lateral, pre}}|) \right| \ge 0.8\text{ meters}$$
6. **Location Constraints**: The maneuver must occur inside the outer roundabout boundary ($r_A < 14.0$ and $r_B < 14.0$).

---

## 3. Code Implementation & Explanation

The logic is implemented in [`src/safety/unsafe_overtaking_rule.py`](../../src/safety/unsafe_overtaking_rule.py).

### Global Heading Vector Calculation
```python
def _track_direction(track: pd.DataFrame) -> Optional[Tuple[float, float]]:
    if len(track) < 2:
        return None
    start = track.iloc[0]
    end = track.iloc[-1]
    direction = (float(end["world_x"]) - float(start["world_x"]),
                 float(end["world_y"]) - float(start["world_y"]))
    if math.hypot(direction[0], direction[1]) < 0.1:
        return None
    return _normalize(direction)
```
* **Explanation**: This extracts a normalized vector representing the overall travel trajectory of a track.

### Relative Projections
```python
# Project the relative position vector onto the average travel heading
rel = (float(row_a["world_x"]) - float(row_b["world_x"]),
       float(row_a["world_y"]) - float(row_b["world_y"]))

forward = _dot(rel, rel_dir)  # Dot product for longitudinal distance
lateral = _cross(rel_dir, rel)  # Cross product for lateral distance
```
* **Explanation**: 
  * The dot product computes how far ahead or behind $A$ is relative to $B$ along their shared direction of travel.
  * The cross product computes the perpendicular side-distance (lane offset).

### Transition Logic Verification
```python
# 1. Check that A started behind and ended up ahead of B
behind_frames = [s for s in sequence if s["forward"] < -BEHIND_AHEAD_THRESHOLD]
ahead_frames = [s for s in sequence if s["forward"] > BEHIND_AHEAD_THRESHOLD]
if not behind_frames or not ahead_frames:
    continue

first_behind = behind_frames[0]["frame"]
last_ahead = ahead_frames[-1]["frame"]
if first_behind >= last_ahead:
    continue

# 2. Check that the distance decreases (approaching) and then increases (passing)
pre_dist = [s["dist"] for s in window if s["frame"] <= crossing_frame]
post_dist = [s["dist"] for s in window if s["frame"] >= crossing_frame]
if not pre_dist or not post_dist or min(pre_dist) >= max(post_dist):
    continue

# 3. Check for a lateral shift (steering out and back in) of at least 0.8 meters
lateral_before = [abs(s["lateral"]) for s in window if s["frame"] <= crossing_frame]
lateral_after = [abs(s["lateral"]) for s in window if s["frame"] >= crossing_frame]
lateral_change = abs(np.median(lateral_after) - np.median(lateral_before))
if lateral_change < LATERAL_MOVEMENT_THRESHOLD:
    continue
```
* **Explanation**: 
  * We verify that the track frame indices follow a strict temporal sequence (behind $\rightarrow$ passing $\rightarrow$ ahead).
  * We verify that the distance converges then diverges (`min(pre_dist) < max(post_dist)`) to rule out cases where the vehicles are simply driving in adjacent lanes at constant spacing.
  * We verify a lateral path deviation threshold of $0.8$ meters to filter out instances of straight-line tailgating or parallel queueing.

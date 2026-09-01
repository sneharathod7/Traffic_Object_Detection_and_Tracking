# Wrong-Way Driving Detection Rule

This document details the algorithm and implementation of the **Wrong-Way Driving Detection Rule** inside the roundabout circulatory ring.

---

## 1. Physical Context & Rule Description
Roundabouts operate under strict directional constraints. In countries with left-hand traffic (such as India), vehicles must navigate the circulatory ring in a **clockwise** direction. Any vehicle navigating in a counter-clockwise direction is driving "Wrong-Way", creating an extreme hazard for incoming and circulating traffic.

The rule states:
> A vehicle is flagged for a wrong-way violation if it travels inside the circulatory ring in a counter-clockwise direction continuously for a minimum of 15 frames (0.5 seconds).

---

## 2. Mathematical Formulation

### A. Coordinate Transformation
Let the center of the roundabout be $(X_c, Y_c) = (43.5, 28.5)$. For any vehicle position $(x_i, y_i)$ at frame $i$, we define the relative coordinates:
$$dx_i = x_i - X_c$$
$$dy_i = y_i - Y_c$$

We convert these to Polar coordinates:
1. **Radial distance ($r_i$)**:
   $$r_i = \sqrt{dx_i^2 + dy_i^2}$$
2. **Polar angle ($\theta_i$)**:
   $$\theta_i = \text{atan2}(dy_i, dx_i)$$
   where $\theta_i \in (-\pi, \pi]$.

### B. Circulatory Ring Restriction
A vehicle is eligible for evaluation only if it resides within the physical boundaries of the roundabout's circulatory ring:
$$R_{\text{MIN}} \le r_i \le R_{\text{MAX}}$$
where $R_{\text{MIN}} = 6.0$ meters (the inner island radius) and $R_{\text{MAX}} = 14.0$ meters (the outer ring boundary).

### C. Shortest Angular Displacement ($\Delta\theta$)
To determine direction of rotation, we compute the displacement $\Delta\theta_i$ between successive frames. Because the angle wraps around at $\pm\pi$, a simple difference $\theta_i - \theta_{i-1}$ would yield massive spike values when crossing the boundary. Instead, we compute the shortest angular path:
$$\Delta\theta_i = \text{atan2}(\sin(\theta_i - \theta_{i-1}), \cos(\theta_i - \theta_{i-1}))$$

### D. Angular Velocity ($\omega$)
The angular velocity $\omega_i$ (in radians per second) is:
$$\omega_i = \Delta\theta_i \times \text{FPS}$$

* **$\omega_i > 0$**: Clockwise movement (correct way).
* **$\omega_i < 0$**: Counter-clockwise movement (wrong way).

We define a negative threshold $\omega_{\text{threshold}} = -0.1$ rad/s to ignore sub-pixel tracking jitter.

---

## 3. Code Implementation & Explanation

The core logic is implemented in [`src/safety/wrong_way_rule.py`](../../src/safety/wrong_way_rule.py).

### Coordinate Conversion & Angle Calculation
```python
# Calculate Polar Radius 'r' and Angle 'theta' relative to center (X_c, Y_c)
df['dx'] = df['world_x'] - X_c
df['dy'] = df['world_y'] - Y_c
df['r'] = np.sqrt(df['dx']**2 + df['dy']**2)
df['theta'] = np.arctan2(df['dy'], df['dx'])
```
* **Explanation**: Shifting the origin to the center of the roundabout allows us to treat rotation as a 1D polar problem.

### Shortest Angular Distance and Velocity
```python
# Compute shortest angular change (resolving wrapping at boundary)
theta_shift = group['theta'].shift(1)
group['delta_theta'] = np.arctan2(
    np.sin(group['theta'] - theta_shift), 
    np.cos(group['theta'] - theta_shift)
)

# Calculate angular velocity (radians per second)
group['omega'] = group['delta_theta'] * FPS
```
* **Explanation**: By using $\sin$ and $\cos$ inside `arctan2`, we project the angular difference onto a unit circle. This automatically selects the shortest arc direction (e.g., $+350^\circ$ difference is correctly resolved as $-10^\circ$).

### Radial Filtering & Direction Classification
```python
# Check if vehicle is inside the circulating ring
group['is_in_ring'] = (group['r'] >= R_MIN) & (group['r'] <= R_MAX)

# Flag wrong way if inside ring and angular velocity is negative
group['is_wrong_way'] = (group['omega'] < OMEGA_THRESHOLD) & group['is_in_ring']
```
* **Explanation**: A binary indicator `is_wrong_way` is created for every frame of a vehicle's track.

### Temporal Persistence Filtering
```python
# Group consecutive True/False blocks to count continuous frames
group['consecutive_group'] = (group['is_wrong_way'] != group['is_wrong_way'].shift()).cumsum()

# Filter wrong-way frames
wrong_way_frames = group[group['is_wrong_way']]

if not wrong_way_frames.empty:
    counts = wrong_way_frames.groupby('consecutive_group').size()
    valid_groups = counts[counts >= CONSECUTIVE_FRAMES_THRESHOLD].index
    
    if not valid_groups.empty:
        # Save first frame of the first violating sequence
        first_valid_group = valid_groups[0]
        violation_start_frame = wrong_way_frames[
            wrong_way_frames['consecutive_group'] == first_valid_group
        ]['frame'].iloc[0]
```
* **Explanation**:
  1. We compute a cumulative sum of changes in `is_wrong_way`. This yields unique group IDs for consecutive blocks of matching boolean states.
  2. We count the size of each `True` block.
  3. If any continuous block is $\ge 15$ frames, it is classified as a valid violation, and the start frame of that block is recorded. This prevents noisy tracker jitter from triggering false alarms.

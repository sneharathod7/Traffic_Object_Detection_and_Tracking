# Pipeline Step 4: Coordinate Mapping (Homography)

## Architectural Overview
Object detection models operate strictly in a 2D pixel coordinate system `[0, W]` and `[0, H]`. However, all traffic safety analysis models (Time-To-Collision, gap acceptance, speeding violation) require inputs in physical metric units (meters, seconds).

The mapping from pixels to meters is handled by `src/homography.py` (`CoordinateMapper` class), which transforms the 2D pixel manifold into a 2D Euclidean metric space.

---

## 1. Scale-Factor Method (Nadir Cameras)
If the drone is hovering exactly overhead pointing straight down (Nadir angle = 0°), the ground plane is perfectly parallel to the camera's image sensor.
In this strict condition, there is zero perspective distortion. 10 pixels at the top of the frame equals exactly the same physical distance as 10 pixels at the bottom of the frame.

We compute a global affine scaling constant:
```python
scale_factor = car_real_length_m / car_pixel_length_px
# Example: 4.0 meters / 55.0 pixels = 0.0727 meters/pixel
```

During execution, the tracking pipeline maps the smoothed coordinates using simple scalar multiplication:
```python
world_x = cx_smooth * scale_factor
world_y = cy_smooth * scale_factor
```

## 2. Full Perspective Homography (Tilted Cameras)
If the drone camera is tilted (e.g. 15° pitch), the scale is no longer uniform. Objects near the top of the frame (closer to the horizon) occupy fewer pixels than objects at the bottom. A uniform scale factor would geometrically compress the velocity of vehicles at the top of the frame.

To solve this, we compute a **3x3 Homography Matrix `H`** using OpenCV (`cv2.findHomography`).
The user supplies at least 4 Ground Control Points (GCPs):
- `pts_src`: The pixel coordinates `(u, v)` of four distinct landmarks (e.g. lane boundary corners).
- `pts_dst`: The physically surveyed meters `(X, Y)` of those same four landmarks relative to an arbitrary origin (e.g., `(0,0)` at the bottom-left corner).

The matrix `H` solves the equation:
```math
\begin{bmatrix} X' \\ Y' \\ W \end{bmatrix} = \mathbf{H} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
```
The true world coordinates are then recovered by perspective division:
```math
X_{world} = X' / W
```
```math
Y_{world} = Y' / W
```
This mathematically flattens the image, perfectly nullifying the camera's tilt and restoring uniform Euclidean distance across the entire scene.

---

## 3. Instantaneous Velocity Calculation
Once the coordinates are mapped to `(world_x, world_y)`, velocity is calculated using the first derivative of position with respect to time.

Because we already applied the `7-frame` Moving Average Smoother to the pixel coordinates in Step 3, the metric coordinates are perfectly stable.

For frame `N`, the distance traveled since frame `N-1` is computed using the Euclidean norm:
```python
delta_x = current_world_x - prev_world_x
delta_y = current_world_y - prev_world_y
distance_m = math.hypot(delta_x, delta_y)

# Velocity = distance / time
velocity_ms = distance_m * video_fps
```
This metric velocity (meters per second) is then attached to the `STrack` dictionary object and passed to the Export module.

# Pipeline Step 3: Post-Processing & Smoothing

## Architectural Overview
Once `tracker.py` finishes the assignment and prediction logic for a single frame, the raw tracks are extremely accurate in identity, but physically unstable. The coordinates suffer from algorithmic jitter, and the class labels suffer from deep-learning scale ambiguity (e.g., small cars classified as motorcycles).

The pipeline invokes two dedicated modules (`postprocess_vehicle_classes.py` and `smoothing.py`) to enforce physical reality.

---

## 1. Algorithmic Vehicle Class Relabeling
Implemented in `src/postprocess_vehicle_classes.py`.
Deep learning models trained on ground-level datasets inherently associate "small bounding box area" with "motorcycle" and "massive area" with "truck". From a drone at 100 meters altitude, *every* car is small.

To fix systemic misclassification, we apply strict geometrical constraints to the final tracked outputs:

### Small Car to Motorcycle Downgrade
If the network assigns a `car` (Class 2) label to a detection, we calculate its pixel dimensions:
`box_length = max((x2 - x1), (y2 - y1))`
If `box_length < 32.0` (configurable via `args.small_car_threshold`), it is physically impossible for the object to be a car based on the established pixel-to-meter scale of the scene. The pipeline forcibly overrides the classification to `motorcycle` (Class 3).

### Heavy Truck to Car Downgrade
In Indian traffic footage, large SUVs and grouping artifacts are frequently mislabeled as `truck` (Class 7).
Because heavy trucks are exceptionally rare in the inner-city intersections we analyze, we apply a hard override:
If `class_id == 7`, the label is forcibly converted to `car` (Class 2) to maintain a standardized 4-class taxonomy (Person, Car, Motorcycle, Bus).

---

## 2. Moving-Average Trajectory Smoothing
Implemented in `src/smoothing.py`.
Directly using raw Kalman filter outputs for downstream safety rules (like Time-To-Collision) is impossible. 
Integer pixel rounding causes the `center_x` coordinate to bounce between `[104, 105, 104]` across 3 frames. If evaluated mathematically, this stationary vehicle registers as moving at `30 km/h` back and forth.

### Rolling Window Algorithm
We apply a temporal Low-Pass Filter using a simple moving average.
For every track ID, the module maintains a fixed-length queue (default `window_size = 7`) of its most recent `(center_x, center_y)` coordinates.

When a new coordinate is received at frame `N`:
1. The new coordinate is pushed to the queue.
2. If the queue length exceeds `7`, the oldest coordinate is popped.
3. The smoothed coordinate `(cx_smooth, cy_smooth)` is calculated as the arithmetic mean of all coordinates in the queue:
```math
cx_{smooth} = \frac{1}{k} \sum_{i=1}^{k} x_i
```

### Delay vs Smoothing Tradeoff
A `7-frame` window at `30 FPS` introduces a mathematical lag of exactly `3 frames` (100 milliseconds). 
For an offline safety analysis tool, a 100ms delay is completely irrelevant, but the resulting 80% reduction in high-frequency spatial noise allows for perfectly smooth trajectory rendering and highly stable velocity derivative calculations in the Homography step.

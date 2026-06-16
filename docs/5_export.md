# Pipeline Step 5: Export (Video & CSV)

## Architectural Overview
The `Exporter` module (`src/export.py`) is the terminus of the pipeline. It transforms the final track data into human-readable visual overlays and machine-readable tabular formats.

Video encoding via OpenCV `VideoWriter` is extremely I/O bound. If executed synchronously inside the main tracker loop, it creates a massive processing bottleneck, dropping the pipeline's overall FPS by up to 50%.

To prevent this, the export pipeline runs heavily optimized **Asynchronous Threading**.

---

## 1. Asynchronous Video Rendering
Implemented using Python's `threading` and `queue.Queue`.

### The Queue Architecture
The main execution loop (`main.py`) acts solely as a Producer. Once a frame finishes the Detection → Tracking → Smoothing → Mapping pipeline, the raw RGB frame and its final list of tracking dictionaries are bundled into a tuple and pushed into an unbounded thread-safe `Queue`.

The `Exporter` module spawns a dedicated Consumer daemon thread upon initialization:
- The thread continuously polls the queue.
- When an item is retrieved, the thread takes over the CPU-intensive process of drawing OpenCV shapes (bounding boxes, text badges, polylines) on the matrix.
- The thread then blocks on the I/O-heavy `VideoWriter.write(frame)` operation.
- Because this occurs in a background thread, the main Tracker loop instantly begins processing frame `N+1` while frame `N` is still being rendered and encoded to disk.

### Visual Overlay Elements
- **Color Palettes**: Uses a hardcoded BGR palette to uniquely color-code the 4 main vehicle classes (e.g. `(0, 165, 255)` Orange for Motorcycles).
- **Transparency (Alpha Blending)**: Bounding boxes and badges are drawn onto an overlay matrix, which is then blended with the original frame via `cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)`. This ensures that dense annotations don't completely obscure the underlying vehicles.
- **Polyline Fade (Trajectory Tail)**: The track's `history` queue (last 40 coordinates) is plotted as a multi-segment line. The thickness and alpha value of each line segment decrease exponentially the further back in time it represents, creating a visual "comet tail" that clearly defines the vehicle's historical path.

---

## 2. CSV Data Export
The primary analytical output is the raw track data, appended frame-by-frame into a CSV file.

### Memory Optimization
Instead of holding 100,000+ track observations in RAM until the video completes, the `Exporter` opens the CSV file in `append` mode and writes the data to disk periodically.

### Schema Definition
The exported CSV is strictly formatted to be instantly parseable by Pandas (`pd.read_csv`).

| Column | Data Type | Implementation Detail |
|---|---|---|
| `frame` | `int32` | 0-indexed video frame counter. |
| `track_id` | `int32` | The `STrack.track_id` integer. Guaranteed to be unique globally across the video. |
| `class_name` | `str` | Final string label post-smoothing (e.g., `"motorcycle"`). |
| `x1, y1, x2, y2` | `float32` | Absolute pixel coordinates of the YOLO/RT-DETR bounding box. |
| `center_x, center_y` | `float32` | Mathematical pixel centers of the bounding box. Subject to the 7-frame Moving Average Smoother. |
| `world_x, world_y` | `float32` | Metric coordinates transformed by `homography.py` (either via Scale Factor or 3x3 Matrix). |
| `confidence` | `float32` | The detector's original probability score `[0.0 - 1.0]`. |
| `velocity_ms` | `float32` | Calculated as `math.hypot(world_dx, world_dy) * video_fps`. |

### Downstream Ingestion
This tabular data structure serves as the direct input to safety-rule evaluation models. By keeping the tracking pipeline entirely decoupled from the safety-rule engine, researchers can run the massive neural networks once to generate the CSV, and then instantly evaluate thousands of different hypothetical safety rules on the structured data without re-running the computer vision stack.

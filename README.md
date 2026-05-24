# Traffic Detection & Tracking Pipeline

> **High-accuracy detection + tracking for dense drone traffic footage**  
> Stationary top-down camera · Indian intersections · Small-object optimised

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Pipeline Explanation](#3-pipeline-explanation)
4. [Design Decisions](#4-design-decisions)
5. [Installation](#5-installation)
6. [How to Run](#6-how-to-run)
7. [Expected Output](#7-expected-output)
8. [Future Improvements](#8-future-improvements)
9. [Safety-Rule Integration (Placeholder)](#9-safety-rule-integration-placeholder)

---

## 1. Project Overview

This project implements a **production-grade, modular pipeline** for detecting and tracking road users in drone-captured traffic videos of Indian intersections. It is designed specifically for:

- **Dense scenes** with 50–100+ simultaneous objects per frame.
- **Small objects** that occupy as few as 15×8 pixels in the raw video.
- **Stationary cameras**, allowing all motion in the scene to be attributed entirely to vehicles and pedestrians.
- **Downstream safety analysis** — every design choice is made to maximise the accuracy and temporal consistency of the output data that feeds rule-based or ML safety models.

### Tracked classes

| Class | COCO ID |
|---|---|
| person | 0 |
| car | 2 |
| motorcycle | 3 |
| bus | 5 |
| truck | 7 |

---

## 2. Problem Statement

Standard off-the-shelf detection + tracking pipelines fail on dense Indian intersection footage for four interconnected reasons:

| Challenge | Root cause | Effect |
|---|---|---|
| Small objects | Drone altitude + YOLO internal downsampling | Cars appear as 10×5 px blobs; detection recall drops below 40% |
| Dense packing | Vehicles touching / overlapping | High IoU between unrelated objects; correct matches rejected |
| Frequent occlusion | Pedestrians behind vehicles, vehicles behind each other | ID switches every 1–2 seconds with naive SORT |
| Misleading pixel speeds | No pixel-to-metre calibration | Velocity-based safety rules fire on false positives |

This pipeline addresses all four problems with a coherent set of design decisions described in [§4](#4-design-decisions).

---

## 3. Pipeline Explanation

```
Input Video
    │
    ▼
┌─────────────────────────────────────────────────┐
│  STEP 1 — DETECTION (detection.py + tiling.py)  │
│                                                 │
│  ┌─────────────┐   ┌──────────────────────────┐ │
│  │ Full frame  │   │ Tiles (2×2 or 3×3 grid)  │ │
│  │ YOLOv8m     │   │ YOLOv8m per tile         │ │
│  │ imgsz=1280  │   │ same imgsz               │ │
│  └──────┬──────┘   └────────────┬─────────────┘ │
│         │                       │               │
│         └───────────┬───────────┘               │
│                     │                           │
│             Class-aware NMS                     │
│                     │                           │
│         List[{bbox, conf, class}]               │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  STEP 2 — BYTETRACK (tracker.py)                │
│                                                 │
│  Stage 1: high-conf dets ↔ all known tracks     │
│  Stage 2: low-conf dets  ↔ unmatched tracks     │
│  Kalman prediction between frames               │
│  Class-aware IoU cost matrix                    │
│                                                 │
│         List[{track_id, bbox, class}]           │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  STEP 3 — SMOOTHING (smoothing.py)              │
│                                                 │
│  Moving-average (window=7) per track            │
│  Removes high-frequency jitter from             │
│  detection instability                          │
│                                                 │
│         smoothed (cx, cy) per track             │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  STEP 4 — COORDINATE MAPPING (homography.py)   │
│                                                 │
│  scale [m/px] = real_car_m / pixel_car_px       │
│  world_x = cx × scale                           │
│  world_y = cy × scale                           │
│  velocity = Δ(world) × fps                      │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  STEP 5 — EXPORT (export.py)                    │
│                                                 │
│  • Annotated MP4 video                          │
│  • Trajectory polyline overlay                  │
│  • CSV: frame, track_id, class,                 │
│         bbox, center_px, world_xy,              │
│         confidence, velocity_ms                 │
└─────────────────────────────────────────────────┘
```

---

## 4. Design Decisions

### 4.1 YOLOv8 — Why this model family?

- Pre-trained on COCO which includes all five target classes.
- Anchor-free architecture handles variable-aspect-ratio objects (motorcycles vs buses).
- Native Python API (`ultralytics`) integrates cleanly with custom pipelines.
- `yolov8m` offers +5–8 mAP over `yolov8n` at only 2× inference time — an excellent accuracy/speed tradeoff for offline batch processing.
- `yolov8l` can be substituted for even higher accuracy when GPU memory allows.

### 4.2 High Input Resolution (imgsz=1280)

YOLO was originally trained at 640×640. Doubling the resolution **halves the effective scale reduction** applied to the input, which directly improves recall for objects smaller than 32×32 pixels. The quadratic increase in FLOPs is offset by the fact that batch-processing a recorded video is not time-critical.

### 4.3 Tiling — Why it helps small objects

YOLO downsamples the input to `imgsz × imgsz`. In a 4K video (3840×2160), a typical car occupies 40×20 pixels = 0.02% of the image area. After downsampling to 1280×1280, that car becomes only 13×7 pixels — often below the network's effective detection threshold.

**With tiling (2×2 grid):**
Each tile covers 1920×1080 pixels, downsampled to 1280×1280, so the same car becomes **~27×13 pixels** — more than double the relative size. This dramatically increases recall for motorcycles and pedestrians, which are even smaller.

The overlap (default 20%) prevents objects straddling a tile boundary from being cut in half. A final class-aware NMS step removes duplicates introduced by the overlap.

### 4.4 ByteTrack — Why it outperforms SORT in crowded scenes

**SORT** associates only high-confidence detections with existing tracks. In a dense intersection, 30–50% of objects are partially occluded at any frame. Their YOLO confidence drops below the threshold, causing SORT to lose them and assign new IDs when they reappear — resulting in hundreds of spurious ID switches per minute.

**ByteTrack** introduces a second association stage:

```
Stage 1: high-conf dets ↔ all tracks  (match clear detections)
Stage 2:  low-conf dets ↔ remaining   (recover occluded objects)
```

Low-confidence detections that would be discarded by SORT instead confirm that an existing track is still present, even if partially occluded. In practice this **reduces ID switches by 30–60%** in dense scenes, which is critical for computing time-to-collision and trajectory-based safety rules.

**Class-aware matching**: The IoU cost between a `car` detection and a `motorcycle` track is set to infinity (no match possible). This prevents the extremely common failure mode where a stationary motorcycle "steals" the ID of a nearby car when the car is temporarily occluded.

### 4.5 Trajectory Smoothing — Why it is critical for safety analysis

Even after Kalman-filter-based tracking, output bounding boxes still carry frame-to-frame jitter from:
- YOLO's stochastic output (same object, slightly different box each frame).
- Quantisation noise from integer pixel rounding.
- Brief partial occlusions.

**Impact on safety metrics without smoothing:**

| Metric | Effect of jitter |
|---|---|
| Speed estimation | A stationary vehicle appears to have speed ~4 m/s (false speeding alarm) |
| TTC (Time-To-Collision) | High variance makes the metric unusable at short horizons |
| Direction of travel | Noisy heading vector; wrong-way detection unreliable |
| Lane occupancy | Object centre oscillates across lane boundary every 3–5 frames |

A 7-frame moving-average window reduces jitter by ~80% while introducing only a 3-frame lag — completely acceptable for safety analysis of recorded footage.

### 4.6 Pixel-to-Metre Conversion — Why pixel distances are misleading

Two vehicles whose pixel centres are 50 px apart could be 2 m apart (near camera) or 20 m apart (far from camera) depending on the drone altitude and tilt. Safety rules are defined in physical units:

- *"Minimum following distance: 3 m"*
- *"Speed limit: 15 m/s (~54 km/h)"*

Without calibration, neither rule can be evaluated. The pipeline supports:

1. **Simple scale** (recommended for nadir/top-down cameras): user provides the pixel length of a reference car. `scale = real_length_m / pixel_length_px`.
2. **Full homography** (for slightly tilted cameras): user annotates 4+ ground-control points with known metric coordinates. OpenCV computes a perspective transform that accounts for spatially-varying scale.

---

## 5. Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (NVIDIA ≥ 8 GB VRAM for 1280px, or 16 GB for 1536px with 3×3 tiling)
- macOS Apple Silicon also supported via MPS backend

### Steps

```bash
# 1. Clone / navigate to the project
cd traffic_tracking

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Place your input video
cp /path/to/intersection.mp4 data/video/
```

The first run will automatically download `yolov8m.pt` (≈50 MB) from the Ultralytics model hub unless you provide a local path.

**Optional — use a larger model for higher accuracy:**
```bash
# Download manually or let ultralytics auto-download on first use
# yolov8l.pt ≈ 87 MB | yolov8x.pt ≈ 136 MB
python -c "from ultralytics import YOLO; YOLO('yolov8l.pt')"
```

---

## 6. How to Run

All commands are run from the `traffic_tracking/src/` directory.

```bash
cd src
```

### Quickstart (recommended defaults)

```bash
python main.py --input ../data/video/intersection.mp4
```

This will:
- Detect and track using `yolov8m.pt` at imgsz=1280 with 2×2 tiling.
- Use a default 0.05 m/px scale (override with `--car-pixel-length`).
- Write annotated video to `../outputs/video/intersection_tracked.mp4`.
- Write CSV to `../outputs/csv/intersection_tracks.csv`.

### Full options example

```bash
python main.py \
    --input  ../data/video/intersection.mp4 \
    --output-video ../outputs/video/tracked.mp4 \
    --output-csv   ../outputs/csv/tracks.csv \
    --model  ../models/yolov8l.pt \
    --imgsz  1280 \
    --conf   0.25 \
    --iou    0.50 \
    --tile-grid 3x3 \
    --tile-overlap 0.20 \
    --high-thresh  0.50 \
    --low-thresh   0.10 \
    --match-thresh 0.80 \
    --track-buffer 30 \
    --min-hits 3 \
    --smooth-window 7 \
    --car-real-length  4.0 \
    --car-pixel-length 55.0 \
    --trajectory-length 40 \
    --device cuda \
    --verbose
```

### Disable tiling (faster, lower recall on small objects)

```bash
python main.py --input ../data/video/intersection.mp4 --tile-grid 1x1
```

### CSV-only mode (no video write overhead)

```bash
python main.py --input ../data/video/intersection.mp4 --no-video
```

### Using a fine-tuned model

```bash
python main.py \
    --input ../data/video/intersection.mp4 \
    --model ../models/yolov8m_finetuned.pt
```

Fine-tuning procedure:
1. Annotate frames using Roboflow or CVAT (YOLO format).
2. Place dataset in `data/annotations/`.
3. Run: `yolo train model=yolov8m.pt data=data.yaml epochs=50 imgsz=1280`
4. Use `runs/detect/train/weights/best.pt` as `--model`.

---

## 7. Expected Output

### Annotated Video (`outputs/video/<name>_tracked.mp4`)

Each frame contains:
- **Colour-coded bounding boxes** (green=person, blue=car, orange=motorcycle, magenta=bus, yellow=truck).
- **Label badge** above each box: `ID:<n> <class>`.
- **Trajectory polyline**: last 40 smoothed positions per track, fading from present (bright) to past (dim).

### CSV (`outputs/csv/<name>_tracks.csv`)

One row per (frame, track_id) pair:

| Column | Type | Description |
|---|---|---|
| `frame` | int | 0-based video frame index |
| `track_id` | int | Persistent unique ID across entire video |
| `class_name` | str | Detected class label |
| `x1, y1, x2, y2` | float | Bounding box pixel coordinates |
| `center_x, center_y` | float | Smoothed box centre (pixels) |
| `world_x, world_y` | float | Centre converted to metres |
| `confidence` | float | YOLO detection score [0, 1] |
| `velocity_ms` | float | Speed estimate in m/s |

### Console summary

```
========================================================
  PROCESSING SUMMARY
========================================================
  frames_processed               1800
  video_fps                      25.0
  resolution                     1920x1080
  total_detections               148320
  avg_dets_per_frame             82.40
  unique_track_ids               312
  processing_fps                 4.2
  scale_factor_m_per_px          0.072727
  coordinate_mode                scale
  output_video                   outputs/video/intersection_tracked.mp4
  output_csv                     outputs/csv/intersection_tracks.csv
========================================================
```

---

## 8. Future Improvements

| Area | Improvement | Expected gain |
|---|---|---|
| **Detection** | Fine-tune YOLOv8l on annotated Indian traffic dataset | +8–15 mAP; far fewer missed pedestrians on 2-wheelers |
| **Detection** | YOLOv9 / RT-DETR as drop-in backend | Better recall at same FPS |
| **Tiling** | Adaptive tiling (denser tiles in high-density image regions) | Fewer redundant tiles in sky/background areas |
| **Tracking** | StrongSORT (ReID embeddings) | Resolves ID switches in camera re-entry scenarios |
| **Tracking** | Per-class Kalman noise parameters | Motorcycles move differently from buses; class-specific motion models |
| **Calibration** | GCP extraction from lane markings | Eliminates manual pixel-length measurement |
| **Calibration** | Fisheye / radial distortion correction | Improves accuracy near frame edges for wide-angle lenses |
| **Performance** | TensorRT engine export for GPU inference | 3–5× speedup, enabling near-real-time processing |
| **Performance** | Batch tile inference (stack tiles into single YOLO call) | 2× speedup on tiling; reduces Python-level overhead |
| **Output** | Heatmap of pedestrian density per zone | Direct input for congestion metrics |
| **Output** | Interactive HTML trajectory viewer (Plotly / Folium) | Easier review of long videos |

---

## 9. Safety-Rule Integration (Placeholder)

The CSV output is specifically designed to feed a downstream safety-rule engine. Planned rules:

```python
# Example rule skeleton — implementation in future sprint
import pandas as pd

df = pd.read_csv("outputs/csv/intersection_tracks.csv")

# Rule 1: Minimum following distance
# Flag frame where two vehicles of same class are < 3 m apart.

# Rule 2: Speed violation
# Flag any vehicle with velocity_ms > SPEED_LIMIT (e.g. 11.1 m/s = 40 km/h).

# Rule 3: Wrong-way driving
# Compute heading from consecutive (world_x, world_y) pairs.
# Compare to expected lane direction from a predefined map.

# Rule 4: Pedestrian in conflict zone
# Flag frames where a 'person' track centre is inside a defined polygon
# (e.g. vehicle travel lane) while a 'car' / 'motorcycle' is approaching.

# Rule 5: Sudden braking
# Detect large negative velocity gradient (deceleration > threshold).
```

The `world_x`, `world_y`, and `velocity_ms` columns — computed by this pipeline — are the primary inputs to all five rules. The smoothing step (§4.5) is a prerequisite for rules 2, 3, 4, and 5 to function reliably.

---

## Project Structure

```
traffic_tracking/
│
├── data/
│   ├── video/            ← Place input .mp4 files here
│   ├── frames/           ← Optional: extracted frames for annotation
│   └── annotations/      ← YOLO-format labels for fine-tuning
│
├── models/               ← Store downloaded / fine-tuned .pt weights
│
├── src/
│   ├── tiling.py         ← Overlapping tile generation + class-aware NMS
│   ├── detection.py      ← YOLOv8 wrapper with tiling support
-  ├── tracker.py        ← ByteTrack implementation + Kalman filter
│   ├── smoothing.py      ← Moving-average and Kalman trajectory smoothers
│   ├── homography.py     ← Pixel-to-metre coordinate mapping
│   ├── export.py         ← CSV writer + annotated video exporter
│   ├── utils.py          ← Logging, colours, drawing helpers
│   └── main.py           ← CLI entry point + pipeline orchestration
│
├── outputs/
│   ├── video/            ← Annotated output videos
│   └── csv/              ← Track data CSV files
│
├── requirements.txt
└── README.md
```

---

*Built for academic research on automated traffic safety analysis at Indian intersections.*

# Setup & Installation Guide

This document provides complete, step-by-step instructions to set up, configure, and execute the **Traffic Object Detection, Tracking, and Automated Safety Analysis** pipeline on your local machine or server.

---

## 1. System Requirements

* **Operating System:** Windows 10/11, Linux (Ubuntu 20.04+), or macOS (CPU mode).
* **Python Version:** Python `3.9`, `3.10`, or `3.11` (Python 3.10 recommended).
* **Hardware Requirements:**
  * **GPU (Recommended):** NVIDIA GPU with $\ge 6\text{ GB}$ VRAM (e.g., RTX 3060/4060 or higher) and CUDA 11.8 / 12.1.
  * **CPU Mode:** Supported, but inference will run at lower frame rates.
  * **RAM:** Minimum $8\text{ GB}$ (16 GB recommended for 4K video processing).

---

## 2. Environment Setup

### Option A: Using Conda (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/sneharathod7/Traffic_Object_Detection_and_Tracking.git
cd Traffic_Object_Detection_and_Tracking

# 2. Create a new conda environment
conda create -n traffic_btp python=3.10 -y
conda activate traffic_btp
```

### Option B: Using Python `venv`

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate
```

---

## 3. Installing Dependencies

### Step 1: Install PyTorch with CUDA Support

Visit [pytorch.org](https://pytorch.org/get-started/locally/) or run the appropriate command for your CUDA version:

```bash
# For CUDA 12.1 (Standard modern NVIDIA GPUs):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU-Only (No NVIDIA GPU):
pip install torch torchvision
```

### Step 2: Install Core Pipeline Dependencies

```bash
# Install core perception, vision, tracking, and SAHI dependencies
pip install -r requirements.txt
```

### Step 3: Install Machine Learning Dependencies

```bash
# Install ML classification and feature extraction libraries
pip install -r "ML Model/requirements.txt"
```

---

## 4. Directory Structure Initialization

Ensure the following default directory layout is present for inputs and outputs:

```bash
# Create required input and output folders
mkdir -p data/video
mkdir -p models
mkdir -p outputs/csv
mkdir -p outputs/video
```

Place your drone input videos (e.g., `intersection.mp4`) inside the `data/` or `data/video/` directory.

---

## 5. Model Weights Setup

The pipeline supports both **RT-DETR (DINO Transformer)** and **YOLOv8** backends.

* **Automatic Download:** Ultralytics and SAHI will automatically download pre-trained weights (`rtdetr-x.pt`, `rtdetr-l.pt`, `yolov8m.pt`) on the first run if they are not found locally.
* **Manual Download (Optional):** You can place custom or pre-downloaded `.pt` weights inside the `models/` directory or project root.

---

## 6. Pipeline Execution & Quickstart Guide

The complete pipeline operates across **3 sequential stages**:

```
[ Raw Video (.mp4) ] ──► [ Stage 1: Detection & Tracking ] ──► [ Trajectory CSV ]
                                                                      │
        ┌─────────────────────────────────────────────────────────────┴──────────────────────────────────────┐
        ▼                                                                                                    ▼
[ Stage 2: Kinematic Safety Rules ]                                                  [ Stage 3: ML Conflict Prediction ]
- Wrong-Way Driving                                                                  - 27-dim Interaction Features
- Safe Space (Tailgating)                                                            - Random Forest Model
- Erratic Lane Weaving                                                               - 5-Tier Risk Alerts (.mp4)
- Unsafe Overtaking / Shortcuts
```

---

### Stage 1: Object Detection & Tracking

Run the core tracking pipeline on your drone video to extract vehicle trajectories and metric coordinates:

```bash
# Standard tracking execution
python src/main.py \
    --input data/intersection.mp4 \
    --output-video outputs/video/tracked.mp4 \
    --output-csv outputs/csv/tracks.csv \
    --conf 0.25 \
    --tile-grid 3x3 \
    --device cuda
```

* **Parameters:**
  * `--input`: Path to input video file.
  * `--output-video`: Destination path for annotated tracking video.
  * `--output-csv`: Destination path for trajectory CSV (contains frame, track_id, bounding boxes, world coordinates, speed).
  * `--tile-grid`: Slicing grid dimensions (e.g., `3x3` or `2x2`).
  * `--device`: `cuda` or `cpu`.

---

### Stage 2: Deterministic Kinematic Safety Rules

Once you have generated the tracking CSV (`data/tracks.csv` or `outputs/csv/tracks.csv`), evaluate specific traffic violations:

#### 1. Wrong-Way Driving Detection:
```bash
python src/safety/wrong_way_rule.py \
    --tracks data/tracks.csv \
    --output outputs/wrong_way.csv

# Generate Wrong-Way Annotated Video:
python src/safety/visualizations/wrong_way_v.py \
    --video data/intersection.mp4 \
    --tracks data/tracks.csv \
    --violations outputs/wrong_way.csv \
    --output outputs/video/wrong_way_annotated.mp4
```

#### 2. Tailgating / Safe-Space Rule:
```bash
python src/safety/safe_space_rule.py \
    --tracks data/tracks.csv \
    --output outputs/tailgating_violations.csv

# Generate Tailgating Annotated Video:
python src/safety/visualizations/visualize_violations.py \
    --video data/intersection.mp4 \
    --tracks data/tracks.csv \
    --rule outputs/tailgating_violations.csv \
    --output outputs/video/tailgating_annotated.mp4
```

#### 3. Erratic Lane Weaving:
```bash
python src/safety/jittering_rule.py \
    --tracks data/tracks.csv \
    --video data/intersection.mp4 \
    --output_csv outputs/output_erratic_weaving.csv \
    --output_video outputs/video/erratic_weaving_annotated.mp4
```

#### 4. Unsafe Overtaking (Trajectory Plot):
```bash
python src/safety/visualizations/visualize_unsafe_overtaking_plot.py \
    data/tracks.csv \
    outputs/unsafe_overtaking_violations.csv \
    outputs/unsafe_overtaking_plot.png
```

#### 5. Unsafe Roundabout Shortcut:
```bash
python src/safety/visualizations/visualize_shortcut_violations.py \
    --video data/intersection.mp4 \
    --tracks data/tracks.csv \
    --shortcuts outputs/unsafe_shortcut_violations.csv \
    --output outputs/video/unsafe_shortcut_annotated.mp4
```

---

### Stage 3: Machine Learning Conflict Prediction

To train or run inference with the **27-dimensional spatiotemporal Random Forest model**:

```bash
cd "ML Model"

# 1. Build master training dataset (merges annotations and tracks)
python step1_build_master.py

# 2. Extract multi-frame interaction features
python step2_feature_engineering.py

# 3. Train Random Forest model, calibrate probabilities, and save model assets
python step3_train_final.py

# 4. Generate 5-Tier dynamic visual danger alert video
python step4_visualize.py
```

* **Output:** The serialized model assets (`danger_model_production.pkl`, `calibrator_production.pkl`) and the annotated risk video (`video/intersection_annotated.mp4`).

---

## 7. Performance & Accuracy Evaluation

To evaluate detection and tracking against ground-truth manual annotations:

```bash
python src/evaluate_metrics.py
```

This will compute **MOTA, IDF1, Precision, Recall, and False Alarm Rates (FPR)** and export `ground_truth/metrics_report.txt`.

---

## 8. Troubleshooting & FAQs

### Q1: `CUDA out of memory` during detection?
* **Solution:** Reduce the batch size or switch to a lighter model variant:
  ```bash
  python src/main.py --input data/intersection.mp4 --model models/rtdetr-m.pt --imgsz 640
  ```

### Q2: OpenCV VideoWriter fails to export `.mp4`?
* **Solution:** Ensure you have the `openh264` or `ffmpeg` codecs installed. On Linux, run:
  ```bash
  sudo apt-get install ffmpeg libsm6 libxext6
  ```

### Q3: How to adapt the roundabout center coordinates for a different location?
* **Solution:** Modify `CENTER_X`, `CENTER_Y`, `R_INNER`, and `R_OUTER` in [`src/safety/calibration.py`](src/safety/calibration.py) or run the interactive calibration utility:
  ```bash
  python src/safety/interactive_calibration.py
  ```

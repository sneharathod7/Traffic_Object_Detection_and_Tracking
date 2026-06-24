# 🚀 Complete Project Run Guide

## Traffic Object Detection & Tracking — Long-Duration Validation

This guide covers everything needed to run the full tracking pipeline and the new validation suite on a powerful machine with long-duration traffic videos.

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GTX 1660 (6GB VRAM) | NVIDIA RTX 3060+ (8GB+ VRAM) |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB free | 100 GB free (for output videos) |
| CPU | 4 cores | 8+ cores |

### Software Requirements

```
Python 3.9+
CUDA 11.8+ (for GPU inference)
Git
```

---

## Step 1: Clone & Setup

```bash
git clone https://github.com/sneharathod7/Traffic_Object_Detection_and_Tracking.git
cd Traffic_Object_Detection_and_Tracking
git checkout testSelfnew
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't cover everything, here are the critical packages:

```bash
# Core
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install opencv-python-headless
pip install numpy pandas scipy tqdm pyyaml

# Detection (SAHI + RT-DETR)
pip install sahi
pip install transformers

# Resource monitoring (optional but recommended)
pip install psutil pynvml
```

---

## Step 2: Prepare Video Data

Place your videos in the `data/video/` directory:

```
data/
└── video/
    ├── clip_5min.mp4       # 5-minute video
    ├── clip_10min.mp4      # 10-minute video
    ├── clip_20min.mp4      # 20-minute video
    └── clip_30min.mp4      # 30-minute video
```

> **Important:** Videos must be aerial/drone footage of Indian traffic at 1920×1080 resolution, 30 FPS. Other resolutions will work but the homography calibration may need adjustment.

---

## Step 3: Run the Tracking Pipeline

### Basic Run (Single Video)

```bash
cd src
python main.py \
    --input ../data/video/clip_5min.mp4 \
    --output-video ../outputs/video/clip_5min_tracked.mp4 \
    --output-csv ../outputs/csv/clip_5min_tracks.csv \
    --device cuda
```

### What This Does

1. **Detection** — SAHI-tiled RT-DETR-X detects vehicles, motorcycles, persons, buses, trucks
2. **Tracking** — BYTETracker with ReID-based association, direction-aware matching, trajectory memory
3. **Recovery** — Velocity extrapolation + conservative Resurrection Layer (score > 0.85)
4. **Smoothing** — Moving average filter on center coordinates
5. **Homography** — Projects pixel coordinates to real-world metric coordinates
6. **Export** — Annotated video + CSV with all track data

### Output Files

```
outputs/
├── video/
│   └── clip_5min_tracked.mp4          # Annotated video with track IDs
├── csv/
│   └── clip_5min_tracks.csv           # Per-frame tracking data
├── metrics/
│   ├── diagnostics_summary.json       # Basic tracking diagnostics
│   ├── switch_log.json                # ID switch events log
│   └── resurrection_log.json          # Resurrection attempt log
└── debug/
    └── motorcycle_failures/           # Debug clips for motorcycle tracking failures
```

### Run Multiple Videos

Run each video separately. The CSV outputs will be used by the validation suite:

```bash
python main.py --input ../data/video/clip_5min.mp4  --output-csv ../outputs/csv/clip_5min_tracks.csv  --device cuda
python main.py --input ../data/video/clip_10min.mp4 --output-csv ../outputs/csv/clip_10min_tracks.csv --device cuda
python main.py --input ../data/video/clip_20min.mp4 --output-csv ../outputs/csv/clip_20min_tracks.csv --device cuda
python main.py --input ../data/video/clip_30min.mp4 --output-csv ../outputs/csv/clip_30min_tracks.csv --device cuda
```

---

## Step 4: Run the Validation Suite

### 4a. Stability Report (Single Video)

```bash
python tracking_stability_report.py \
    --csv ../outputs/csv/clip_5min_tracks.csv \
    --fps 30 \
    --resurrection-log ../outputs/metrics/resurrection_log.json \
    --switch-log ../outputs/metrics/switch_log.json \
    --window-minutes 1 \
    --output-dir ../outputs/validation
```

> **Window size guidance:**
> - 5-minute video → use `--window-minutes 1`
> - 10-minute video → use `--window-minutes 2`
> - 20-minute video → use `--window-minutes 5`
> - 30-minute video → use `--window-minutes 5`

**Outputs:**
- `outputs/validation/tracking_stability_report.json` — Full structured report
- `outputs/validation/tracking_stability_report.md` — Human-readable markdown

### 4b. Stability Report (Batch — All Videos)

After running the tracker on all videos, analyze them together:

```bash
python tracking_stability_report.py \
    --csv-dir ../outputs/csv/ \
    --fps 30 \
    --window-minutes 5 \
    --output-dir ../outputs/validation
```

**Outputs:**
- Per-video: `clip_5min_tracks_stability.json`, `clip_5min_tracks_stability.md`, etc.
- Combined: `outputs/validation/combined_tracking_report.json`

### 4c. Drift Analysis

```bash
python tracking_drift_analysis.py \
    --report ../outputs/validation/tracking_stability_report.json \
    --output-dir ../outputs/validation
```

**Outputs:**
- `outputs/validation/tracking_drift_analysis.json`
- `outputs/validation/tracking_drift_analysis.md`

### 4d. Resource Monitoring (Wrap a Full Run)

```bash
python resource_monitor.py \
    --command "python main.py --input ../data/video/clip_20min.mp4 --output-csv ../outputs/csv/clip_20min_tracks.csv --device cuda" \
    --output ../outputs/validation/resource_usage.json \
    --poll-interval 2.0
```

**Output:**
- `outputs/validation/resource_usage.json` — FPS, peak RAM, peak GPU memory, GPU utilization

---

## Step 5: Interpret the Results

### Tracking Health Score (THS)

| Score Range | Grade | Meaning |
|-------------|-------|---------|
| 80–100 | 🟢 Excellent | Production-ready |
| 60–79 | 🟡 Acceptable | Minor issues, monitor closely |
| 0–59 | 🔴 Needs Improvement | Not deployment-ready |

### Identity Continuity Score (ICS)

| Score Range | Grade | Meaning |
|-------------|-------|---------|
| 80–100 | 🟢 Strong | Identity preservation is reliable |
| 60–79 | 🟡 Moderate | Some identity breaks, investigate fragmentation |
| 0–59 | 🔴 Weak | Frequent identity failures |

### Drift Analysis Verdict

| Status | Meaning |
|--------|---------|
| 🟢 STABLE | Quality consistent across the video — safe for long runs |
| 🟢 IMPROVING | Tracker warms up and gets better over time |
| 🟡 SLIGHTLY_DEGRADING | One metric declining — monitor on longer videos |
| 🔴 DEGRADING | Multiple metrics declining — investigate before production |

---

## Step 6: Run the Merge Audit (Optional Verification)

If you want to verify that tracked identities are clean (no false merges):

```bash
python merge_audit.py \
    --csv ../outputs/csv/clip_5min_tracks.csv \
    --video ../data/video/clip_5min.mp4
```

Look for:
- **Suspicious tracks with score > 0.40** → Potential false merges
- **Appearance jumps** → Possible identity corruption
- **Velocity jumps** → Possible teleportation errors

---

## What I Need From You

After running the pipeline on long-duration videos, share the following files so I can analyze the results:

### Priority 1 (Essential)

| File | Why |
|------|-----|
| `tracking_stability_report.json` | Complete metrics for each video |
| `tracking_drift_analysis.json` | Drift classification for each video |
| `combined_tracking_report.json` | Cross-video comparison (if batch mode) |

### Priority 2 (Very Useful)

| File | Why |
|------|-----|
| `resurrection_log.json` | How many resurrections attempted/succeeded |
| `switch_log.json` | Where ID switches are happening |
| `resource_usage.json` | FPS and memory footprint for deployment planning |

### Priority 3 (Nice to Have)

| File | Why |
|------|-----|
| `diagnostics_summary.json` | Basic tracking diagnostics |
| The `.md` reports | Human-readable summaries you can read directly |
| One tracked video clip (30s sample) | Visual verification of tracking quality |

---

## Quick Reference — All Commands in Order

```bash
# 1. Setup
git checkout testSelfnew
pip install -r requirements.txt

# 2. Run tracker
cd src
python main.py --input ../data/video/YOUR_VIDEO.mp4 \
    --output-csv ../outputs/csv/YOUR_VIDEO_tracks.csv \
    --device cuda

# 3. Stability report
python tracking_stability_report.py \
    --csv ../outputs/csv/YOUR_VIDEO_tracks.csv \
    --fps 30 \
    --resurrection-log ../outputs/metrics/resurrection_log.json \
    --switch-log ../outputs/metrics/switch_log.json \
    --window-minutes 5 \
    --output-dir ../outputs/validation

# 4. Drift analysis
python tracking_drift_analysis.py \
    --report ../outputs/validation/tracking_stability_report.json \
    --output-dir ../outputs/validation

# 5. Resource monitoring (optional — wraps the tracker run)
python resource_monitor.py \
    --command "python main.py --input ../data/video/YOUR_VIDEO.mp4 --output-csv ../outputs/csv/YOUR_VIDEO_tracks.csv --device cuda" \
    --output ../outputs/validation/resource_usage.json

# 6. Merge audit (optional — verifies no false merges)
python merge_audit.py \
    --csv ../outputs/csv/YOUR_VIDEO_tracks.csv \
    --video ../data/video/YOUR_VIDEO.mp4
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce `--imgsz` to 960 or use `--device cpu` (slower) |
| `ModuleNotFoundError: sahi` | `pip install sahi` |
| `ModuleNotFoundError: psutil` | `pip install psutil` (only needed for resource monitoring) |
| Very low FPS (<0.3) | Normal for SAHI-tiled detection on 1080p. Use a faster GPU |
| Tracker produces too many IDs | Check that `config.yaml` has `resurrection.enabled: true` |
| JSON serialization error | Already fixed — ensure you're on branch `testSelfnew` |

---

## Config Reference

The tracker is configured via `config.yaml`. **Do NOT change these values** — the tracker is frozen as the validated baseline:

```yaml
# Key settings (DO NOT MODIFY)
resurrection:
  enabled: true
  score_threshold: 0.85        # Conservative — only high-confidence recoveries
  motorcycle_app_threshold: 0.9 # Extra strict for motorcycles
  max_gap: 20                   # Max frames a lost track can be resurrected

association:
  weights:
    iou: 0.3
    appearance: 0.2
    direction: 0.2
    trajectory: 0.15
    motion: 0.1
    scale: 0.05
```

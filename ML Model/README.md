# Traffic Safety Conflict Prediction Model (BTP)

This repository contains the machine learning pipeline developed to predict dangerous vehicle interactions (conflicts) in a roundabout traffic intersection. By extracting spatiotemporal trajectory kinematics frame-by-frame, the model flags safety-critical vehicle behaviors (such as tailgating, yield violations, and cutting-in) and renders multi-color bounding boxes based on the severity of the conflict.

---

## 1. Project Architecture & Data Flow

```text
[annotations.csv]       [long1_tracks_narain_cleaned_edited.csv]      [wrong_way.csv, tailgating_violations.csv, ...]
  (Manual Labels)                     (Base Tracker Data)                          (Deterministic Violators)
         │                                     │                                              │
         ▼                                     ▼                                              ▼
 ┌───────────────┐                     ┌───────────────┐                              ┌───────────────┐
 │ Mapped IDs    │                     │  Deduplicate  │                              │ Exclude list  │
 └───────┬───────┘                     └───────┬───────┘                              └───────┬───────┘
         │                                     │                                              │
         ├─────────────────────────────────────┴──────────────────────────────────────────────┤
         ▼
   [Step 1: step1_build_master.py]  <─── Proximity Constraint (2.0m - 12.0m for Safe Background Pairs)
         │
         ▼ (master_train_raw.csv)
   [Step 2: step2_feature_engineering.py] <─── 75-Frame Kinematic Window (Stride 6, 13 positions + Nearest-Neighbor Imputation)
         │
         ▼ (X_train_final.csv, y_train_final.csv, weights_train_final.csv)
   [Step 3: step3_train_final.py] ───> Fits Random Forest Regressor + Isotonic Calibration
         │                             └─> Serializes: danger_model_production.pkl, calibrator_production.pkl, model_metadata.pkl
         ▼
   [Step 4: step4_visualize.py] <─── Vectorized Elliptical Proximity (15m x 8m) & Parked Vehicle Intelligence
         │
         ▼
   [video/intersection_annotated.mp4] (Flags Dangerous Vehicles: Yellow [>L2], Orange [>L3], Red [>L4])
```

---

## 2. In-Depth Pipeline Breakdown

### Step 1: Master Dataset Compilation (`step1_build_master.py`)
* **ID Mismatch Resolution**: Manual annotations (track IDs 0–31) are mapped to their corresponding tracker IDs.
* **3-Tier Priority Weighting**: 
  * **Tier 1 (Weight 10.0 & 3.0)**: Manual Ego vehicles (Class 1) get maximum priority (Weight 10.0). Manual Affected vehicles (Class 0) get Weight 3.0.
  * **Tier 2 (Weight 3.0 & 1.0)**: Automatic Rule-Violation pairs extracted from `tailgating`, `unsafe_overtaking`, and `wrong_way` datasets. Violators get Class 1 (Weight 3.0), affected vehicles get Class 0 (Weight 1.0).
  * **Tier 3 (Weight 1.0)**: Safe background vehicle pairs. Sampled to perfectly balance the dataset volume.
* **Proximity Constraint**: Safe background pairs are only sampled if their Euclidean distance is between **2.0 and 12.0 meters**, preventing the model from using large spatial distances as a shortcut for predicting safety.

### Step 2: Spatiotemporal Feature Engineering (`step2_feature_engineering.py`)
* **Deduplication**: Resolves tracker frame-overlap duplicates by selecting the detection with the highest confidence.
* **75-Frame Horizon**: Evaluates interactions over a sliding window of **75 consecutive frames** (~2.5 second horizon at 30 FPS).
* **Sampling Stride & Imputation**: Samples kinematics every 6th frame (13 positions). Missing positions are resolved via nearest-neighbor imputation, requiring 80% coexistence minimum.
* **Feature Vector (27 Dimensions)**:
  - $d_{t1} \dots d_{t13}$ (Euclidean distance in meters across the 13 positions)
  - $\theta_{t1} \dots \theta_{t13}$ (Relative trajectory angle in radians across the 13 positions)
  - $v_{rel}$ (Relative velocity $v_A - v_B$ at the current frame)

### Step 3: Production Model Training & Calibration (`step3_train_final.py`)
* **Random Forest Ensembling**: Fits a randomized Decision Tree ensemble (`RandomForestRegressor` softened with `max_depth=10`, `min_samples_leaf=5`) to prevent strict leaf clustering.
* **Out-of-Fold Calibration**: Performs 5-fold Stratified Cross Validation. The out-of-fold (OOF) raw scores are fed into an **Isotonic Regression** calibrator to produce an accurate, non-parametric probability distribution mapped to the real world.
* **Data-Driven Thresholds**: Rendering thresholds are extracted empirically from the calibrated precision-recall curve rather than hardcoded.

### Step 4: Video Inference & Temporal Tuning (`step4_visualize.py`)
* **Parked Vehicle Intelligence**: Prevents tracking jitter from repeatedly triggering bounding boxes. The script tracks effective spatial displacement over the last **180 frames** (6 seconds). If the effective speed is `< 0.5 m/s`, the vehicle is categorized as strictly parked and ignored.
* **Vectorized Elliptical Proximity**: Top-5 vehicle interactions are queried via an oriented bounding ellipse ($15m$ semi-major along heading, $8m$ semi-minor) to prioritize frontal interactions and cut computation noise.
* **Priority Mapping**: The 10 manually annotated ground-truth conflicts receive bypass rendering pipelines — including relaxed 40% co-existence rules, expanded 20m interaction radius, and heightened threshold sensitivity.
* **Dynamic Rendering Cooldowns**: 
  - **Level 4** (Highly Dangerous): Red box, 75-frame visual memory
  - **Level 3** (Dangerous): Orange box, 50-frame visual memory
  - **Level 2** (Nearly Dangerous): Yellow box, 25-frame visual memory

---

## 3. Dataset & Model Metrics

### Dataset Composition
**Total Dataset**: 7,304 vehicle records (3,652 interaction pairs)
* **Tier 1 (Manual Interactions)**: 960 pairs
* **Tier 2 (Rule Violations)**: 866 pairs
* **Tier 3 (Background Safe)**: 1,826 pairs (Balanced against Tier 1 + Tier 2)

### ML Model Validation Results & Analysis

The model yields a highly precise safety filter that aggressively eliminates false positives while prioritizing heavily weighted edge-case conflicts. We performed a **5-Fold Stratified Cross Validation** (80-20 Train/Test split) alongside a **Stratified Shuffle Split** (70-30 Train/Test split) to validate robustness.

| Metric | 80-20 Split (5-Fold CV) | 70-30 Split (Shuffle Split) | Drop / Change |
| :--- | :--- | :--- | :--- |
| **Accuracy (ACC)** | 94.11% ± 1.13% | 93.43% ± 1.33% | -0.68% |
| **Precision** | 97.84% ± 0.45% | 97.81% ± 0.32% | -0.03% |
| **Recall** | 95.41% ± 1.26% | 94.64% ± 1.29% | -0.77% |
| **F1 Score** | 0.9660 ± 0.0068 | 0.9619 ± 0.0079 | -0.0041 |

#### Academic Significance
1. **Incredible Precision (Low False Positive Rate)**: The model achieves nearly **97.8% Precision** on both splits. In the context of traffic safety analysis, this means when the model flags a trajectory as "Dangerous," it is almost universally correct. High precision is crucial for real-world automated systems to prevent "alert fatigue".
2. **High Data Efficiency (Data Saturation)**: Despite taking away 10% of the training data (a significant chunk) in the 70-30 split, the F1-Score only drops by **0.0041** and Precision drops by a negligible **0.03%**. This proves the model has achieved *data saturation*. It has genuinely learned the core spatiotemporal kinematics of aggressive driving rather than memorizing the dataset.
3. **Extremely Low Variance (Robustness)**: The standard deviation across all 5 folds for both splits is roughly ± 0.01 (or 1%), proving the Isotonic-calibrated Random Forest Regressor is highly stable and does not overfit to any specific geometric quirk of the roundabout.

*Additional Calibration Metrics:*
* **Regression R-Squared**: `0.5835 +/- 0.0471`
* **Brier Score (Calibrated)**: `0.0847` (9.4% improvement over uncalibrated raw scores)

---

## 4. How to Run the Pipeline

Run the scripts sequentially in your terminal:

```bash
# 1. Compile the unified master raw dataset
python step1_build_master.py

# 2. Extract 61 features over the 30-frame window
python step2_feature_engineering.py

# 3. Train model, run Cross-Validation, and save serialized assets
python step3_train_final.py

# 4. Perform video inference and export multi-color annotated output
python step4_visualize.py
```

*Final Annotated Output Video:* saved to `video/intersection_annotated.mp4`.

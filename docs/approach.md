# Architecture Enhancements: From SAHI+DETR to SAHI+DINO

This document theoretically outlines the algorithmic and structural enhancements applied to the traffic detection and tracking pipeline to resolve severe track fragmentation (ID switching) and class flickering in highly dense Indian traffic scenarios.

---

## 1. Detection Backend: DINO Integration (RT-DETRv2-X)

**Problem:** The previous `RT-DETR-L` model exhibited fluctuating confidence scores and missed small, heavily occluded vehicles (like motorcycles squeezed between cars).
**Enhancement:** Upgraded the detection backbone to `rtdetr-x.pt` (RT-DETRv2 Extra-Large).
**Theoretical Benefit:** This model incorporates **DINO (Improved deNoising Optimization)** mechanics and multi-scale attention. It handles dense object distributions significantly better by refining bounding box coordinates iteratively, resulting in highly stable confidence scores across consecutive frames—a strict prerequisite for stable ByteTrack matching.

---

## 2. Tile Collision Mitigation: Weighted Box Fusion (WBF)

**Problem:** Because Indian traffic lacks lane discipline, vehicles constantly straddle SAHI slice boundaries. SAHI's standard Non-Maximum Suppression (NMS) relies purely on IoU. If a motorcycle on a tile boundary is detected twice with sub-pixel misalignment, the IoU drops drastically (e.g., to 0.28). Standard NMS fails to suppress the duplicate, passing two bounding boxes for one motorcycle to the tracker, triggering an instant ID switch.
**Enhancement:** Implemented a custom **Weighted Box Fusion** algorithm directly into the detection output stream.
**Theoretical Benefit:**

1. **Spatial Clustering:** First groups detections by physical center-point proximity (e.g., within 30 pixels).
2. **Confident Suppression:** Within each cluster, the highest-confidence bounding box acts as the anchor. Lower-confidence boxes are suppressed using a significantly relaxed IoU threshold.
   _Result:_ Eliminated the "ghost detections" at tile edges without accidentally suppressing adjacent, closely-packed vehicles.

---

## 3. Tracker Dynamics: Scale-Proportional Kalman Noise

**Problem:** The internal ByteTrack `STrack` instances were initialized with **static, absolute** OpenCV Kalman Filter noise matrices (e.g., a measurement noise of exactly `5.0` pixels). In 1080p drone footage, where bounding boxes range from 50 to 300 pixels, static noise causes the Kalman Filter to become "infinitely confident" in its predictions. If a vehicle abruptly braked or swerved, the predicted trajectory missed the actual detection, dropping the track completely and assigning a new ID.
**Enhancement:** Replaced static values with **Dynamic Size-Based Noise Scaling**.
**Theoretical Benefit:** Process and measurement noise are now calculated dynamically at every frame as a function of the bounding box height and width (e.g., `noise = 0.05 * max(w, h)`). This exactly mirrors the mathematics of the original DeepSORT/ByteTrack whitepapers, ensuring the tracker's uncertainty region expands appropriately for large vehicles and remains tight for small ones.

---

## 4. Class Flickering: Soft-Matching & Majority Voting

**Problem:** Strict `class_aware` matching was enabled. If the DINO detector misclassified a motorcycle as a car for even a single frame, the tracker aggressively prevented the new "car" detection from matching the "motorcycle" track. The track was killed, generating massive ID fragmentation.
**Enhancement:**

1. **Soft Penalty Matching:** Disabled strict class-blocking. Instead, cross-class matching is allowed but receives a soft penalty (IoU reduced by 0.2). The tracker heavily prefers same-class objects but can bridge a 1-frame misclassification if the spatial overlap (IoU) is undeniable.
2. **Temporal Majority Voting:** Every track now maintains a `collections.Counter` of its detected classes over its entire lifetime.
   **Theoretical Benefit:** Even if a vehicle oscillates between `car` and `motorcycle` in the raw detection stream, the tracker mathematically smooths the output by always broadcasting the statistical mode (most frequent class). This guarantees stable visual output and clean CSV data.

---

## 5. Visual Interpretability

**Problem:** Verbose tracking outputs (e.g., printing `(STABLE)`, ages, and confidence histories) rendered the video unusable during dense traffic surges.
**Enhancement:** Introduced a `--clean-draw` pipeline flag.
**Theoretical Benefit:** Decouples diagnostic tracking data from visual representation. Relies entirely on color-coding (Magenta = New, Orange = Stable, Cyan = Recovered) to communicate complex state transitions without polluting the visual field with text, allowing human reviewers to easily spot structural ID failures.

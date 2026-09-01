# Traffic Safety Rules Documentation

This directory contains comprehensive, detailed documentation for all traffic safety violation detection rules implemented in the project. Each rule includes physical context, mathematical formulation, parameter settings, and detailed explanations of the key Python code snippets.

## Implemented Safety Rules

### 1. [Wrong-Way Driving Detection Rule](./wrong_way_rule.md)
* **Goal**: Detect vehicles traveling in the wrong direction (counter-clockwise) within the circulatory ring of the roundabout.
* **Key Math**: Polar coordinate transformation, shortest-angular path displacement, and temporal noise filtering.

### 2. [Safe Space Rule (Tailgating)](./safe_space_rule.md)
* **Goal**: Detect vehicles following too closely behind leading vehicles inside the same lane (Proximity Violation).
* **Key Math**: Radial classification, relative polar-angle ordering, and arc-length calculation ($d = R_{\text{avg}} \times \Delta\theta$).

### 3. [Unsafe Overtaking Detection Rule](./unsafe_overtaking_rule.md)
* **Goal**: Identify vehicles executing high-risk overtaking maneuvers within the restricted roundabout zone.
* **Key Math**: Heading vector alignment, spatial projections (forward dot product and lateral cross product), distance convergence/divergence checks, and lateral offset analysis.

### 4. [Unsafe Roundabout Shortcut Rule](./unsafe_roundabout_shortcut_rule.md)
* **Goal**: Detect vehicles cutting across corners at the intersection instead of traveling properly around the central island.
* **Key Math**: Compass-based entry/exit verification, phase-unwrapped angular displacement ($\Delta\theta_{\text{unwrapped}}$), and intersection-level congestion mapping.

### 5. [Erratic Lane Weaving Rule](./erratic_weaving_rule.md)
* **Goal**: Detect erratic lane weaving (`jittering_rule.py`) by tracking physical lane boundary crosses inside the roundabout.
* **Key Math**: Temporal tracking of radial distance $r$, mapping discrete lane states (Inner vs Outer), and accumulating state transitions over a sliding time window.

---

## Constants Reference
Below is the common spatial frame of reference used across all safety rules:

* **Roundabout Center ($X_C, Y_C$)**: $(43.5, 28.5)$
* **Inner Island Radius ($R_{\text{INNER}}$)**: $6.0$ meters
* **Outer Circulatory Boundary ($R_{\text{OUTER}}$)**: $14.0$ meters
* **Video FPS**: $30.0$ frames per second (time step $\Delta t = 1/30$ seconds)

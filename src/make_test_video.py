"""
make_test_video.py — Generate a minimal synthetic test video.

Creates a 30-frame, 1280x720 black video with two white rectangles that
simulate moving cars.  Good enough to exercise the detector pipeline without
needing a real drone clip.

Usage (from traffic_tracking/src/):
    python make_test_video.py
Output:
    ../data/video/test_synthetic.mp4
"""
import cv2
import numpy as np
from pathlib import Path

OUT = Path("../data/video/test_synthetic.mp4")
OUT.parent.mkdir(parents=True, exist_ok=True)

W, H, FPS, N = 1280, 720, 25, 30

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUT), fourcc, FPS, (W, H))

for i in range(N):
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    # "car" 1 — moves right
    x1 = 100 + i * 15
    cv2.rectangle(frame, (x1, 200), (x1 + 80, 240), (200, 200, 200), -1)
    # "car" 2 — moves left
    x2 = 1100 - i * 15
    cv2.rectangle(frame, (x2, 400), (x2 + 80, 440), (180, 180, 180), -1)
    writer.write(frame)

writer.release()
print(f"Test video written -> {OUT.resolve()}")

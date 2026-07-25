"""
main.py — Traffic Tracking Pipeline Entry Point

Pipeline overview (executed per frame):
  1. Read frame from video.
  2. Detector.detect()       → tiled YOLOv8 detections.
  3. BYTETracker.update()    → stable track IDs.
  4. Smoother.update()       → jitter-reduced centre coordinates.
  5. CoordinateMapper.to_world() → metric (x, y) positions.
  6. Exporter.process_frame()→ write annotated video + CSV row.

Usage examples
--------------
Basic run (auto-download yolov8m.pt):
    python main.py --input data/video/intersection.mp4

Full options:
    python main.py \\
        --input  data/video/intersection.mp4 \\
        --output-video outputs/video/tracked.mp4 \\
        --output-csv   outputs/csv/tracks.csv \\
        --model  models/yolov8m.pt \\
        --imgsz  1280 \\
        --conf   0.25 \\
        --tile-grid 3x3 \\
        --smooth-window 7 \\
        --car-real-length 4.0 \\
        --car-pixel-length 55.0 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
from tqdm import tqdm

from detection import Detector
from export import Exporter
from homography import CoordinateMapper
from smoothing import MovingAverageSmoother
from tracker import BYTETracker, STrack
from utils import ensure_dir, format_stats, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="traffic_tracker",
        description=(
            "High-accuracy detection + tracking pipeline for dense drone "
            "traffic footage. Outputs an annotated video and a CSV with "
            "per-frame track data including real-world coordinates."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- I/O ----------------------------------------------------------------
    io = p.add_argument_group("Input / Output")
    io.add_argument(
        "--input", "-i", required=True,
        help="Path to the input video file.",
    )
    io.add_argument(
        "--output-video", "-ov",
        default=None,
        help="Path for the annotated output video (.mp4). "
             "Defaults to outputs/video/<input_stem>_tracked.mp4.",
    )
    io.add_argument(
        "--output-csv", "-oc",
        default=None,
        help="Path for the output CSV. "
             "Defaults to outputs/csv/<input_stem>_tracks.csv.",
    )
    io.add_argument(
        "--no-video", action="store_true",
        help="Skip writing the annotated video (CSV only).",
    )
    io.add_argument(
        "--log-file", default=None,
        help="Optional path to write log messages to a file.",
    )
    io.add_argument(
        "--clean-draw", action="store_true",
        help="Use clean drawing mode (just ID and color) to reduce clutter in dense traffic.",
    )

    # ---- Model / detection --------------------------------------------------
    det = p.add_argument_group("Detection")
    det.add_argument(
        "--detector",
        choices=["yolov8", "sahi_rtdetr", "sahi_dino"],
        default="sahi_rtdetr",
        help=(
            "Detector architecture to use. "
            "'sahi_dino' uses a DINO-style RT-DETRv2-X backend with "
            "Weighted Box Fusion for improved tile-collision handling."
        ),
    )
    det.add_argument(
        "--model", default=None,
        help="Model weights file (defaults: rtdetr-l.pt for sahi_rtdetr, yolov8m.pt for yolov8).",
    )
    det.add_argument("--imgsz",  type=int,   default=1280, help="YOLO input image size.")
    det.add_argument("--conf",   type=float, default=0.10, help="Detection confidence threshold.")
    det.add_argument("--iou",    type=float, default=0.50, help="NMS IoU threshold.")
    det.add_argument(
        "--tile-grid", default="6x6",
        help="Tiling grid as ROWSxCOLS (e.g. '2x2' or '3x3') for YOLOv8. Use '1x1' to disable tiling.",
    )
    det.add_argument("--tile-overlap", type=float, default=0.60,
                     help="Fractional overlap between adjacent tiles (0.0–0.75) for YOLOv8.")
    det.add_argument("--tta", action="store_true",
                     help="Enable test-time augmentation for YOLOv8 (slower, more accurate).")
    det.add_argument("--slice-height", type=int, default=384,
                     help="SAHI slice height.")
    det.add_argument("--slice-width", type=int, default=384,
                     help="SAHI slice width.")
    det.add_argument("--overlap-height-ratio", type=float, default=0.75,
                     help="SAHI slice overlap height ratio.")
    det.add_argument("--overlap-width-ratio", type=float, default=0.75,
                     help="SAHI slice overlap width ratio.")
    det.add_argument("--device", default=None,
                     help="Inference device: 'cuda', 'cpu', 'mps', or '0' for GPU index.")

    # ---- Vehicle Class Postprocessing --------------------------------------
    vpost = p.add_argument_group("Vehicle Class Postprocessing")
    vpost.add_argument("--truck-conf-thresh", type=float, default=0.55,
                       help="Confidence threshold for truck detections.")
    vpost.add_argument("--truck-min-area", type=float, default=8000.0,
                       help="Minimum bounding box area (in pixels) for trucks.")
    vpost.add_argument("--truck-min-width", type=float, default=120.0,
                       help="Minimum bounding box width (in pixels) for trucks.")
    vpost.add_argument("--truck-min-height", type=float, default=50.0,
                       help="Minimum bounding box height (in pixels) for trucks.")
    vpost.add_argument("--bus-conf-thresh", type=float, default=0.30,
                       help="Confidence threshold for bus detections.")
    vpost.add_argument("--car-conf-thresh", type=float, default=0.25,
                       help="Confidence threshold for car/relabel detections.")
    vpost.add_argument("--car-max-motorcycle-area", type=float, default=1600.0,
                       help="Max pixel area for a car before it gets relabeled as a motorcycle.")
    vpost.add_argument("--car-max-motorcycle-width", type=float, default=40.0,
                       help="Max pixel width for a car before it gets relabeled as a motorcycle.")
    vpost.add_argument("--car-max-motorcycle-height", type=float, default=40.0,
                       help="Max pixel height for a car before it gets relabeled as a motorcycle.")
    vpost.add_argument("--person-conf-thresh", type=float, default=0.15,
                       help="Confidence threshold for person detections.")
    vpost.add_argument("--motorcycle-conf-thresh", type=float, default=0.10,
                       help="Confidence threshold for motorcycle detections.")
    vpost.add_argument("--debug-trucks", action="store_true",
                       help="Enable saving cropped truck detections for manual debugging.")

    # ---- Tracking -----------------------------------------------------------
    trk = p.add_argument_group("Tracking (ByteTrack)")
    trk.add_argument("--high-thresh",  type=float, default=0.50,
                     help="Minimum score to use a detection in Stage-1 matching and "
                          "to initialise new tracks.")
    trk.add_argument("--low-thresh",   type=float, default=0.10,
                     help="Minimum score for Stage-2 (occlusion recovery) matching.")
    trk.add_argument("--match-thresh", type=float, default=0.80,
                     help="Max IoU-distance to accept a Stage-1 match (0.8 → IoU ≥ 0.2).")
    trk.add_argument("--track-buffer", type=int,   default=60,
                     help="Frames a Lost track is kept before deletion (60 @ 25 fps = 2.4 s).")
    trk.add_argument("--motorcycle-track-buffer", type=int, default=120,
                     help="Frames a Lost motorcycle track is kept before deletion.")
    trk.add_argument("--motorcycle-match-thresh", type=float, default=0.70,
                     help="Max distance threshold to accept a Stage-1 motorcycle match.")
    trk.add_argument("--min-hits",     type=int,   default=3,
                     help="Consecutive frames before a new track appears in output.")
    trk.add_argument("--class-aware", action=argparse.BooleanOptionalAction, default=True,
                     help="Enable class-aware tracking to prevent cross-class ID stealing.")

    # ---- Smoothing ----------------------------------------------------------
    smo = p.add_argument_group("Smoothing")
    smo.add_argument("--smooth-window", type=int, default=7,
                     help="Moving-average window size (frames). Use 1 to disable.")

    # ---- Coordinate mapping -------------------------------------------------
    cmap = p.add_argument_group("Pixel-to-Metre Mapping")
    cmap.add_argument("--scale-factor", type=float, default=None,
                      help="Direct metres-per-pixel scale factor. "
                           "Overrides --car-real-length / --car-pixel-length.")
    cmap.add_argument("--car-real-length",  type=float, default=4.0,
                      help="Typical car length in metres (reference object).")
    cmap.add_argument("--car-pixel-length", type=float, default=None,
                      help="Measured pixel length of a typical car in the video. "
                           "If not provided, defaults to a 0.05 m/px scale factor.")

    # ---- Visualisation ------------------------------------------------------
    vis = p.add_argument_group("Visualisation")
    vis.add_argument("--no-trajectories", action="store_true",
                     help="Disable trajectory polyline overlay.")
    vis.add_argument("--trajectory-length", type=int, default=40,
                     help="Number of past positions to draw per track.")

    # ---- Misc ---------------------------------------------------------------
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after this many frames (useful for testing).")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG-level logging.")

    return p


# ---------------------------------------------------------------------------
# Helper: parse tile grid string "RxC" → (R, C)
# ---------------------------------------------------------------------------

def parse_tile_grid(s: str):
    try:
        parts = s.lower().split("x")
        return int(parts[0]), int(parts[1])
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Invalid tile-grid format '{s}'. Expected ROWSxCOLS, e.g. '2x2'."
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    """
    Execute the full tracking pipeline and return a summary statistics dict.
    """
    # ---- Logging ------------------------------------------------------------
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level, log_file=args.log_file)

    # ---- Resolve output paths -----------------------------------------------
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input video not found: %s", input_path)
        sys.exit(1)

    stem = input_path.stem

    output_video = None
    if not args.no_video:
        output_video = args.output_video or f"outputs/video/{stem}_tracked.mp4"
        ensure_dir(str(Path(output_video).parent))

    output_csv = args.output_csv or f"outputs/csv/{stem}_tracks.csv"
    ensure_dir(str(Path(output_csv).parent))

    # ---- Open video ---------------------------------------------------------
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", input_path)
        sys.exit(1)

    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total_frames = min(total_frames, args.max_frames)

    logger.info(
        "Video: %s  |  %dx%d @ %.1f fps  |  %d frames",
        input_path.name, width, height, fps, total_frames,
    )

    # ---- Resolve model path based on detector -------------------------------
    model_path = args.model
    if model_path is None:
        if args.detector == "sahi_dino":
            model_path = "rtdetr-x.pt"       # DINO-style Extra-Large
        elif args.detector == "sahi_rtdetr":
            model_path = "rtdetr-l.pt"
        else:
            model_path = "yolov8m.pt"

    # ---- Build pipeline components ------------------------------------------
    if args.detector == "sahi_dino":
        from sahi_dino_detection import SahiDinoDetector
        detector = SahiDinoDetector(
            model_path           = model_path,
            slice_height         = args.slice_height,
            slice_width          = args.slice_width,
            overlap_height_ratio = args.overlap_height_ratio,
            overlap_width_ratio  = args.overlap_width_ratio,
            conf                 = args.conf,
            device               = args.device,
        )
    elif args.detector == "sahi_rtdetr":
        from sahi_rtdetr_detection import SahiRTDetrDetector
        detector = SahiRTDetrDetector(
            model_path           = model_path,
            slice_height         = args.slice_height,
            slice_width          = args.slice_width,
            overlap_height_ratio = args.overlap_height_ratio,
            overlap_width_ratio  = args.overlap_width_ratio,
            conf                 = args.conf,
            device               = args.device,
        )
    else:
        # YOLOv8 with optional tiling
        tile_grid = parse_tile_grid(args.tile_grid)
        use_tiling = tile_grid != (1, 1)
        detector = Detector(
            model_path   = model_path,
            imgsz        = args.imgsz,
            conf         = args.conf,
            iou          = args.iou,
            use_tiling   = use_tiling,
            tile_grid    = tile_grid,
            tile_overlap = args.tile_overlap,
            use_tta      = args.tta,
            device       = args.device,
        )

    # Reset ByteTrack ID counter for reproducibility between runs
    STrack.reset_id_counter()
    tracker = BYTETracker(
        high_thresh  = args.high_thresh,
        low_thresh   = args.low_thresh,
        match_thresh = args.match_thresh,
        track_buffer = args.track_buffer,
        min_hits     = args.min_hits,
        class_aware  = args.class_aware,
        motorcycle_track_buffer = args.motorcycle_track_buffer,
        motorcycle_match_thresh = args.motorcycle_match_thresh,
        device       = args.device,
    )

    smoother = MovingAverageSmoother(window=args.smooth_window)

    if args.scale_factor:
        mapper = CoordinateMapper.from_scale_factor(args.scale_factor)
    elif args.car_pixel_length:
        mapper = CoordinateMapper.from_reference_object(
            real_length_m   = args.car_real_length,
            pixel_length_px = args.car_pixel_length,
        )
    else:
        logger.warning(
            "No pixel-to-metre calibration provided. "
            "Using default 0.05 m/px. Pass --car-pixel-length or --scale-factor "
            "for accurate world coordinates."
        )
        mapper = CoordinateMapper.from_scale_factor(0.05)

    # ---- Statistics counters ------------------------------------------------
    total_detections  = 0
    total_track_ids   = set()
    class_counts: dict = {}
    t_start = time.perf_counter()

    postprocess_stats = {
        "car_detections": 0,
        "truck_detections": 0,
        "truck_to_car_relabels": 0,
        "truck_filtered_out": 0,
        "raw_truck_predictions": 0,
        "bus_detections": 0,
        "bus_filtered_out": 0,
        "person_detections": 0,
        "motorcycle_detections": 0,
    }

    # ---- Main loop ----------------------------------------------------------
    with Exporter(
        fps               = fps,
        output_video_path = output_video,
        output_csv_path   = output_csv,
        frame_size        = (width, height),
        draw_trajectories = not args.no_trajectories,
        trajectory_length = args.trajectory_length,
        clean_draw        = args.clean_draw,
    ) as exporter:

        with tqdm(total=total_frames, unit="frame", desc="Tracking") as pbar:
            frame_idx = 0
            while frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # 1. Detect
                detections = detector.detect(frame)

                # Apply class-aware postprocessing
                from postprocess_vehicle_classes import postprocess_vehicle_detections
                detections = postprocess_vehicle_detections(
                    detections           = detections,
                    frame                = frame,
                    frame_idx            = frame_idx,
                    truck_conf_thresh    = args.truck_conf_thresh,
                    truck_min_area       = args.truck_min_area,
                    truck_min_width      = args.truck_min_width,
                    truck_min_height     = args.truck_min_height,
                    bus_conf_thresh      = args.bus_conf_thresh,
                    car_conf_thresh      = args.car_conf_thresh,
                    car_max_motorcycle_area = args.car_max_motorcycle_area,
                    car_max_motorcycle_width = args.car_max_motorcycle_width,
                    car_max_motorcycle_height = args.car_max_motorcycle_height,
                    person_conf_thresh   = args.person_conf_thresh,
                    motorcycle_conf_thresh = args.motorcycle_conf_thresh,
                    debug_trucks         = args.debug_trucks,
                    stats                = postprocess_stats,
                )

                total_detections += len(detections)

                # 2. Track
                tracks = tracker.update(detections, frame)

                # 3. Smooth + map coordinates
                enriched = []
                smoother.tick()
                for t in tracks:
                    cx, cy = t["center"]
                    s_cx, s_cy = smoother.update(t["track_id"], cx, cy)
                    wx, wy = mapper.to_world(s_cx, s_cy)

                    total_track_ids.add(t["track_id"])
                    class_counts[t["class_name"]] = (
                        class_counts.get(t["class_name"], 0) + 1
                    )

                    enriched.append({
                        **t,
                        "smoothed_cx": s_cx,
                        "smoothed_cy": s_cy,
                        "world_x":     wx,
                        "world_y":     wy,
                    })

                # 4. Export
                exporter.process_frame(frame_idx, frame, enriched)

                frame_idx += 1
                pbar.update(1)
                pbar.set_postfix(
                    dets=len(detections),
                    tracks=len(tracks),
                    ids=len(total_track_ids),
                )

    cap.release()

    elapsed = time.perf_counter() - t_start

    # Run post-tracking diagnostics automatically
    try:
        from diagnostics import run_diagnostics
        logger.info("Running post-tracking diagnostics...")
        run_diagnostics(
            csv_path=Path(output_csv),
            annotations_dir=Path("../data/annotations"),
            output_metrics_dir=Path("../outputs/metrics")
        )
    except Exception as e:
        logger.warning("Could not complete diagnostics run (%s)", e)

    raw_trucks = postprocess_stats.get("raw_truck_predictions", 0)
    trucks_filtered_out = postprocess_stats.get("truck_filtered_out", 0)
    trucks_relabeled = postprocess_stats.get("truck_to_car_relabels", 0)
    total_trucks_filtered = trucks_filtered_out + trucks_relabeled
    filtered_pct = (total_trucks_filtered / raw_trucks * 100.0) if raw_trucks > 0 else 0.0

    stats = {
        "frames_processed":      frame_idx,
        "video_fps":             round(fps, 2),
        "resolution":            f"{width}x{height}",
        "total_detections":      total_detections,
        "avg_dets_per_frame":    round(total_detections / max(frame_idx, 1), 2),
        "unique_track_ids":      len(total_track_ids),
        "class_observation_counts": class_counts,
        "postprocessing_statistics": {
            "car_detections":        postprocess_stats.get("car_detections", 0),
            "truck_detections":      postprocess_stats.get("truck_detections", 0),
            "truck_to_car_relabels": postprocess_stats.get("truck_to_car_relabels", 0),
            "truck_filtered_percentage": f"{filtered_pct:.2f}% (raw: {raw_trucks}, filtered: {trucks_filtered_out}, relabeled: {trucks_relabeled})",
        },
        "processing_time_s":     round(elapsed, 2),
        "processing_fps":        round(frame_idx / max(elapsed, 1e-6), 2),
        "scale_factor_m_per_px": round(mapper.scale_factor, 6),
        "coordinate_mode":       mapper.mode,
        "output_video":          output_video or "disabled",
        "output_csv":            output_csv,
    }
    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Traffic Detection & Tracking Pipeline")
    print("=" * 60)

    stats = run(args)

    print(format_stats(stats))
    print("\nDone. Review the annotated video and CSV for results.\n")


if __name__ == "__main__":
    main()

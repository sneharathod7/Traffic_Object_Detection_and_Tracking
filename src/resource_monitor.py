"""
resource_monitor.py — Runtime Resource Monitoring for Tracking Pipeline

Captures system resource utilization during long tracking runs:
  - Processing FPS
  - Peak RAM usage
  - Peak GPU memory (if CUDA available)
  - Average GPU utilization (if nvidia-smi available)

Usage:
  This module is designed to be used as a context manager wrapping the
  tracking pipeline execution:

    from resource_monitor import ResourceMonitor

    monitor = ResourceMonitor(poll_interval=1.0)
    monitor.start()
    # ... run tracking pipeline ...
    monitor.stop()
    monitor.save("outputs/validation/resource_usage.json")

  Or as a standalone wrapper:

    python resource_monitor.py --command "python src/main.py --input video.mp4" \
        --output outputs/validation/resource_usage.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """
    Background resource monitor that samples system metrics at a fixed interval.

    Captures:
      - CPU usage percentage (per-process if psutil available, else system-wide)
      - RAM usage (current, peak)
      - GPU memory usage and utilization (if NVIDIA GPU + pynvml/nvidia-smi available)
      - Wall-clock elapsed time
    """

    def __init__(self, poll_interval: float = 1.0):
        self.poll_interval = poll_interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_time: float = 0.0
        self._end_time: float = 0.0

        # Collected samples
        self._ram_samples: List[float] = []      # MB
        self._cpu_samples: List[float] = []       # %
        self._gpu_mem_samples: List[float] = []   # MB
        self._gpu_util_samples: List[float] = []  # %

        # Capabilities
        self._has_psutil = False
        self._has_gpu = False
        self._process = None

        try:
            import psutil
            self._has_psutil = True
            self._process = psutil.Process(os.getpid())
        except ImportError:
            logger.warning("psutil not installed. RAM/CPU monitoring will be limited.")

        try:
            import pynvml
            pynvml.nvmlInit()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._has_gpu = True
            self._pynvml = pynvml
            logger.info("GPU monitoring enabled via pynvml.")
        except Exception:
            # Fall back to nvidia-smi
            self._has_gpu = self._check_nvidia_smi()
            if self._has_gpu:
                logger.info("GPU monitoring enabled via nvidia-smi (fallback).")
            else:
                logger.info("No GPU monitoring available.")

    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available on the system."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _sample_nvidia_smi(self) -> Dict[str, float]:
        """Query nvidia-smi for GPU memory and utilization."""
        result = {}
        try:
            mem_out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if mem_out.returncode == 0:
                parts = mem_out.stdout.strip().split(",")
                if len(parts) >= 3:
                    result["gpu_mem_used_mb"] = float(parts[0].strip())
                    result["gpu_mem_total_mb"] = float(parts[1].strip())
                    result["gpu_utilization_pct"] = float(parts[2].strip())
        except Exception:
            pass
        return result

    def _poll_loop(self):
        """Background polling thread."""
        while not self._stop_event.is_set():
            try:
                # RAM
                if self._has_psutil:
                    import psutil
                    mem_info = self._process.memory_info()
                    ram_mb = mem_info.rss / (1024 * 1024)
                    self._ram_samples.append(ram_mb)

                    cpu_pct = self._process.cpu_percent(interval=None)
                    self._cpu_samples.append(cpu_pct)
                else:
                    # Fallback: system-wide memory
                    try:
                        import psutil
                        vm = psutil.virtual_memory()
                        self._ram_samples.append(vm.used / (1024 * 1024))
                    except ImportError:
                        pass

                # GPU
                if self._has_gpu:
                    if hasattr(self, "_pynvml"):
                        pynvml = self._pynvml
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                        self._gpu_mem_samples.append(mem_info.used / (1024 * 1024))

                        util_info = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                        self._gpu_util_samples.append(float(util_info.gpu))
                    else:
                        gpu_data = self._sample_nvidia_smi()
                        if "gpu_mem_used_mb" in gpu_data:
                            self._gpu_mem_samples.append(gpu_data["gpu_mem_used_mb"])
                        if "gpu_utilization_pct" in gpu_data:
                            self._gpu_util_samples.append(gpu_data["gpu_utilization_pct"])

            except Exception as e:
                logger.debug("Resource sampling error: %s", e)

            self._stop_event.wait(self.poll_interval)

    def start(self):
        """Start the background monitoring thread."""
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("Resource monitoring started (interval=%.1fs).", self.poll_interval)

    def stop(self):
        """Stop the background monitoring thread."""
        self._end_time = time.time()
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped. %d samples collected.", len(self._ram_samples))

    def get_report(self, frames_processed: int = 0) -> Dict[str, Any]:
        """Generate a resource usage report from collected samples."""
        elapsed = self._end_time - self._start_time if self._end_time > self._start_time else 0.0

        report: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "platform": {
                "os": platform.system(),
                "os_version": platform.version(),
                "python_version": platform.python_version(),
                "cpu_count": os.cpu_count(),
            },
            "elapsed_seconds": round(elapsed, 2),
            "samples_collected": len(self._ram_samples),
        }

        # FPS
        if frames_processed > 0 and elapsed > 0:
            report["processing_fps"] = round(frames_processed / elapsed, 4)
            report["frames_processed"] = frames_processed
        else:
            report["processing_fps"] = None
            report["frames_processed"] = frames_processed

        # RAM
        if self._ram_samples:
            report["ram"] = {
                "peak_mb": round(max(self._ram_samples), 2),
                "avg_mb": round(sum(self._ram_samples) / len(self._ram_samples), 2),
                "min_mb": round(min(self._ram_samples), 2),
                "final_mb": round(self._ram_samples[-1], 2),
            }
        else:
            report["ram"] = None

        # CPU
        if self._cpu_samples:
            report["cpu"] = {
                "peak_pct": round(max(self._cpu_samples), 2),
                "avg_pct": round(sum(self._cpu_samples) / len(self._cpu_samples), 2),
            }
        else:
            report["cpu"] = None

        # GPU Memory
        if self._gpu_mem_samples:
            report["gpu_memory"] = {
                "peak_mb": round(max(self._gpu_mem_samples), 2),
                "avg_mb": round(sum(self._gpu_mem_samples) / len(self._gpu_mem_samples), 2),
                "min_mb": round(min(self._gpu_mem_samples), 2),
                "final_mb": round(self._gpu_mem_samples[-1], 2),
            }
        else:
            report["gpu_memory"] = None

        # GPU Utilization
        if self._gpu_util_samples:
            report["gpu_utilization"] = {
                "peak_pct": round(max(self._gpu_util_samples), 2),
                "avg_pct": round(sum(self._gpu_util_samples) / len(self._gpu_util_samples), 2),
                "min_pct": round(min(self._gpu_util_samples), 2),
            }
        else:
            report["gpu_utilization"] = None

        return report

    def save(self, output_path: str, frames_processed: int = 0):
        """Save the resource usage report to a JSON file."""
        report = self.get_report(frames_processed)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Resource usage saved to %s", out)
        return report


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone Mode: Monitor an External Command
# ═══════════════════════════════════════════════════════════════════════════════

def monitor_command(command: str, output_path: str, poll_interval: float = 2.0) -> Dict:
    """
    Run a shell command and monitor resource usage during execution.

    This is useful for wrapping `python src/main.py ...` to capture
    resource usage during a full tracking run.
    """
    monitor = ResourceMonitor(poll_interval=poll_interval)
    monitor.start()

    logger.info("Running command: %s", command)
    start_time = time.time()

    try:
        result = subprocess.run(
            command, shell=True,
            capture_output=False,  # Let stdout/stderr pass through
        )
        exit_code = result.returncode
    except KeyboardInterrupt:
        exit_code = -1
        logger.warning("Command interrupted by user.")
    except Exception as e:
        exit_code = -2
        logger.error("Command failed: %s", e)

    elapsed = time.time() - start_time
    monitor.stop()

    report = monitor.get_report(frames_processed=0)
    report["command"] = command
    report["exit_code"] = exit_code

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Resource monitoring complete. Report saved to %s", out)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="resource_monitor",
        description="Monitor resource usage during tracking pipeline execution.",
    )
    parser.add_argument("--command", type=str, required=True,
                        help="Shell command to execute and monitor.")
    parser.add_argument("--output", type=str, default="outputs/validation/resource_usage.json",
                        help="Path to save resource usage JSON.")
    parser.add_argument("--poll-interval", type=float, default=2.0,
                        help="Polling interval in seconds.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    report = monitor_command(args.command, args.output, args.poll_interval)

    print(f"\n{'=' * 60}")
    print(f"  RESOURCE USAGE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Elapsed Time:    {report['elapsed_seconds']:.1f}s")
    if report.get("ram"):
        print(f"  Peak RAM:        {report['ram']['peak_mb']:.0f} MB")
        print(f"  Avg RAM:         {report['ram']['avg_mb']:.0f} MB")
    if report.get("gpu_memory"):
        print(f"  Peak GPU Memory: {report['gpu_memory']['peak_mb']:.0f} MB")
    if report.get("gpu_utilization"):
        print(f"  Avg GPU Util:    {report['gpu_utilization']['avg_pct']:.1f}%")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

import gc
import logging
import time
from pathlib import Path
from typing import Optional

from prometheus_client import Histogram

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)

# Prometheus Metrics
TASK_DURATION_SECONDS = Histogram(
    "task_duration_seconds",
    "Time spent performing the task",
    ["task_name"]
)

TASK_CPU_USAGE_PERCENT = Histogram(
    "task_cpu_usage_percent",
    "CPU usage percent during the task",
    ["task_name"]
)

TASK_MEMORY_USAGE_BYTES = Histogram(
    "task_memory_usage_bytes",
    "Memory usage in bytes at the end of the task",
    ["task_name"]
)


class PerformanceMonitor:
    """Helper to measure Time, CPU, and Memory usage."""

    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0
        self.start_cpu = 0.0
        self.end_cpu = 0.0
        self.start_mem = 0
        self.end_mem = 0
        self.process = psutil.Process() if HAS_PSUTIL else None

    def start(self):
        gc.collect()  # Clean up before starting measurement
        self.start_time = time.time()
        if self.process:
            self.process.cpu_percent(interval=None)  # Set baseline for next call
            self.start_mem = self.process.memory_info().rss

    def stop(self):
        self.end_time = time.time()
        if self.process:
            self.end_cpu = self.process.cpu_percent(interval=None)  # Get average CPU usage since last call
            self.end_mem = self.process.memory_info().rss

    @property
    def duration(self):
        return self.end_time - self.start_time

    def report(self, label: str, count: Optional[int] = None) -> str:
        count_str = f" (N={count})" if count is not None else ""
        msg = f"[{label}]{count_str} Time: {self.duration:.4f}s"
        
        # Record Time
        TASK_DURATION_SECONDS.labels(task_name=label).observe(self.duration)

        if self.process:
            mem_diff_mb = (self.end_mem - self.start_mem) / (1024 * 1024)
            end_mem_mb = self.end_mem / (1024 * 1024)
            msg += f" | CPU: {self.end_cpu:.1f}% | Mem: {end_mem_mb:.1f}MB (Delta: {mem_diff_mb:+.2f}MB)"

            # Record CPU & Memory
            TASK_CPU_USAGE_PERCENT.labels(task_name=label).observe(self.end_cpu)
            TASK_MEMORY_USAGE_BYTES.labels(task_name=label).observe(self.end_mem)

        # Write to log file
        try:
            log_dir = Path("logs/performance")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "performance_log.txt"

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {msg}\n")
        except Exception as e:
            # Just log error but don't fail main flow
            logging.getLogger(__name__).error(f"Failed to write performance log: {e}")

        return msg

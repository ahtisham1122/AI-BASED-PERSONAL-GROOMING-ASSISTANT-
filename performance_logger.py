"""
Performance/stress-test measuring tool for the FYP report (sections 6.8
and 6.9). Sits on top of the real app — it only reads numbers that
main.py already computes/exposes, it never changes face shape / skin
tone / AR behavior.

Used from main.py like this:
    perf_logger = PerformanceLogger()
    ...inside the per-frame loop...
    perf_logger.tick(fps_ema, analyzing, get_last_inference_ms())
    ...in the key-handling code...
    if key == ord('l'):
        perf_logger.save_log()

Can also be run directly (`python performance_logger.py`) to just
(re)create the stress_test_log.txt checklist without starting the app.
"""
import os
import time
from datetime import datetime

import psutil


# ── Stress test checklist (Section 6.9) ──────────────────────────────
STRESS_CASES = [
    "No face in view -> should not crash, should show a prompt",
    "Sideways face -> frame should be skipped",
    "Very dark or very bright light -> skin tone should show low confidence or stay empty",
    "Two faces in frame -> should use the largest face only",
    "Rapid key presses (pressing N, P, D quickly) -> should not crash",
    "Long run, about 30 minutes -> memory and speed should stay stable",
    "Model files removed/renamed -> app should fall back to rule-based mode with a visible badge",
]


def write_stress_test_log(path='stress_test_log.txt'):
    """Creates the pre-filled stress test checklist if it doesn't already exist (never overwrites your notes)."""
    if os.path.exists(path):
        return
    lines = ["AI Grooming Assistant - Stress Test Log", "=" * 40, ""]
    for i, case in enumerate(STRESS_CASES, 1):
        lines.append(f"Case {i}: {case}")
        lines.append("  Result (PASS/FAIL): ")
        lines.append("  Notes: ")
        lines.append("")
    with open(path, 'w') as f:
        f.write("\n".join(lines))
    print(f"[perf] Created {path}")


def _avg(samples):
    return sum(samples) / len(samples) if samples else 0.0


class PerformanceLogger(object):
    """
    Tracks FPS / CNN inference time / analysis-to-lock time / memory,
    prints a live summary every 5 seconds, and can save the current
    averages to performance_results.txt on demand.
    """

    # How often to re-check memory and app.log size — every frame
    # would be wasteful (both touch the OS), a few seconds is plenty
    # for numbers that change slowly.
    MEMORY_CHECK_SECONDS = 3
    SUMMARY_SECONDS      = 5

    def __init__(self, app_log_path='app.log'):
        self._process = psutil.Process(os.getpid())
        self._app_log_path = app_log_path

        self._fps_samples       = []
        self._inference_samples = []
        self._lock_time_samples = []
        self.mem_mb             = 0.0
        self.error_count        = 0

        # Edge-detection for the "analyzing" flag, to time how long a
        # fresh analysis takes to lock (i.e. it goes True -> False).
        self._was_analyzing   = None
        self._analysis_start  = None

        now = time.perf_counter()
        self._last_mem_check = now
        self._last_summary   = now
        self._last_log_size  = self._app_log_size()

        write_stress_test_log()

    def _app_log_size(self):
        try:
            return os.path.getsize(self._app_log_path)
        except FileNotFoundError:
            return 0

    def tick(self, fps, analyzing, inference_ms):
        """Call once per camera frame with the current FPS, whether analysis is still running, and the latest CNN inference time (ms, 0.0 if none yet)."""
        now = time.perf_counter()
        self._fps_samples.append(fps)
        if inference_ms:
            self._inference_samples.append(inference_ms)
        self._track_lock_time(analyzing, now)

        if now - self._last_mem_check > self.MEMORY_CHECK_SECONDS:
            self.mem_mb = self._process.memory_info().rss / (1024 * 1024)
            self._check_app_log_growth()
            self._last_mem_check = now

        if now - self._last_summary > self.SUMMARY_SECONDS:
            print(f"FPS: {_avg(self._fps_samples):.0f} | "
                  f"Inference: {_avg(self._inference_samples):.0f}ms | "
                  f"Memory: {self.mem_mb:.0f}MB")
            self._last_summary = now

    def _track_lock_time(self, analyzing, now):
        if self._was_analyzing is None:
            self._was_analyzing = analyzing
            if analyzing:
                self._analysis_start = now
            return
        if analyzing and not self._was_analyzing:
            self._analysis_start = now  # a fresh analysis just started
        elif not analyzing and self._was_analyzing and self._analysis_start is not None:
            self._lock_time_samples.append((now - self._analysis_start) * 1000)
            self._analysis_start = None
        self._was_analyzing = analyzing

    def _check_app_log_growth(self):
        # ponytail: comparing file size (not parsing tracebacks) is a
        # crude "did something new get logged" check, good enough to
        # flag a crash happened during stress testing; parse app.log
        # directly if an exact error count is ever needed.
        size = self._app_log_size()
        if size > self._last_log_size:
            self.error_count += 1
            print(f"[perf] app.log grew -- error #{self.error_count} logged, check app.log for the traceback")
        self._last_log_size = size

    def save_log(self, path='performance_results.txt'):
        """Appends the current averages, with a timestamp, to performance_results.txt (call on the 'L' key)."""
        with open(path, 'a') as f:
            f.write(f"[{datetime.now().isoformat(timespec='seconds')}]\n")
            f.write(f"  Avg FPS: {_avg(self._fps_samples):.1f}\n")
            f.write(f"  Avg CNN inference time: {_avg(self._inference_samples):.1f} ms\n")
            f.write(f"  Avg analysis-to-lock time (~30-frame average): {_avg(self._lock_time_samples):.0f} ms\n")
            f.write(f"  Memory (RSS): {self.mem_mb:.1f} MB\n")
            f.write(f"  app.log growth events this session: {self.error_count}\n\n")
        print(f"[perf] Saved current averages to {path}")


def _selftest():
    """One runnable check for the non-trivial bit here: the analysis-start/lock edge detection."""
    perf = PerformanceLogger.__new__(PerformanceLogger)  # skip __init__'s file/psutil side effects
    perf._was_analyzing = None
    perf._analysis_start = None
    perf._lock_time_samples = []

    t = 0.0
    perf._track_lock_time(True, t)   # analysis starts
    t += 0.5
    perf._track_lock_time(True, t)   # still analyzing
    t += 0.5
    perf._track_lock_time(False, t)  # locks -> should record ~1000ms
    assert len(perf._lock_time_samples) == 1
    assert 990 <= perf._lock_time_samples[0] <= 1010, perf._lock_time_samples

    t += 0.2
    perf._track_lock_time(True, t)   # new analysis starts again
    assert perf._analysis_start == t
    print("[perf] selftest passed")


if __name__ == "__main__":
    write_stress_test_log()
    _selftest()
    print("[perf] performance_logger.py ready. Run `python main.py` to start measuring live.")

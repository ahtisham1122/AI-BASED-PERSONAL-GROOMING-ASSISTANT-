"""
One-off: measures the new UI's per-frame rendering cost, so we can
check the "no more than 2-3 FPS drop" requirement without needing a
live webcam. Compares:
  1. Baseline: no UI drawing at all (just the resize/copy every frame does).
  2. Cold cache: panel/top/bottom rebuilt every frame (worst case).
  3. Warm cache: state unchanged frame to frame (typical case — the
     panel only actually changes a few times per second in real use).
"""
import cv2
import glob
import time

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # project root
from app.paths import DATASET_DIR
from app.ui import UIRenderer, build_ui_state, _fit_frame, VIDEO_W, VIDEO_H

N = 300
frame = cv2.imread(glob.glob(str(DATASET_DIR / 'testing_set' / 'Oval' / '*'))[0])
frame = cv2.resize(frame, (640, 480))

glasses_names = ["aviator", "round", "square", "wayfarer"]


def make_state(i):
    return build_ui_state(
        gender='male', voice_status='on', fps=30.0,
        face_shape='Oval', face_shape_confidence=70.0 + (i % 5),  # forces cache to change every frame
        analyzing=False, analysis_progress=(30, 30), near_edge=False, cnn_busy=(i % 10 == 0),
        skin_tone='Medium', skin_swatch_rgb=(178, 140, 110),
        glasses_idx=0, glasses_names=glasses_names, suggested=[0, 1], total_glasses=4,
    )


# 1) Baseline: just the letterbox resize, no UI at all
t0 = time.perf_counter()
for i in range(N):
    _ = _fit_frame(frame, VIDEO_W, VIDEO_H)
t1 = time.perf_counter()
baseline_ms = (t1 - t0) / N * 1000
print(f"Baseline (resize only):      {baseline_ms:.3f} ms/frame  (~{1000/baseline_ms:.0f} FPS ceiling)")

# 2) Cold cache: force a rebuild every frame by changing confidence each time
renderer = UIRenderer()
t0 = time.perf_counter()
for i in range(N):
    state = make_state(i)
    _ = renderer.render(frame, state)
t1 = time.perf_counter()
cold_ms = (t1 - t0) / N * 1000
print(f"Cold cache (rebuild/frame):  {cold_ms:.3f} ms/frame  (~{1000/cold_ms:.0f} FPS ceiling)")

# 3) Warm cache: identical state every frame (the realistic case once
# the analysis lock has settled and skin tone/glasses aren't changing)
renderer2 = UIRenderer()
fixed_state = make_state(0)
t0 = time.perf_counter()
for i in range(N):
    _ = renderer2.render(frame, fixed_state)
t1 = time.perf_counter()
warm_ms = (t1 - t0) / N * 1000
print(f"Warm cache (state unchanged):{warm_ms:.3f} ms/frame  (~{1000/warm_ms:.0f} FPS ceiling)")

print(f"\nOverhead vs baseline: cold +{cold_ms - baseline_ms:.3f} ms, warm +{warm_ms - baseline_ms:.3f} ms")
print(f"At a 30fps camera (33.3ms budget), warm-cache overhead costs "
      f"~{(warm_ms - baseline_ms) / (1000/30) * 30:.2f} FPS")

# 4) Realistic mix: panel changes ~every 5th frame (matches the CNN's
# ~5-frame submission cadence during analysis, or skin tone's 15-frame
# cadence), warm the rest of the time.
renderer3 = UIRenderer()
t0 = time.perf_counter()
for i in range(N):
    state = make_state(i // 5)  # only actually changes every 5 frames
    _ = renderer3.render(frame, state)
t1 = time.perf_counter()
mixed_ms = (t1 - t0) / N * 1000
print(f"\nRealistic mix (panel changes every 5th frame): {mixed_ms:.3f} ms/frame "
      f"(~{1000/mixed_ms:.0f} FPS ceiling, +{mixed_ms - baseline_ms:.3f} ms vs baseline, "
      f"~{(mixed_ms - baseline_ms) / (1000/30) * 30:.2f} FPS drop at 30fps)")

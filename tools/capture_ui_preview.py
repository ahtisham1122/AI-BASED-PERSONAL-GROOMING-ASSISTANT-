"""
One-off: exercises the new UIRenderer against a few different states
(analyzing, locked-high-confidence, near-edge) using a real photo as
the stand-in camera frame, and saves data/debug/ui_preview.jpg.
"""
import cv2
import glob
import os

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # project root
from app.paths import DATASET_DIR, DATA_DIR
from app.ui import UIRenderer, build_ui_state

DEBUG_DIR = DATA_DIR / 'debug'
os.makedirs(DEBUG_DIR, exist_ok=True)

frame = cv2.imread(glob.glob(str(DATASET_DIR / 'testing_set' / 'Oval' / '*'))[0])
frame = cv2.resize(frame, (640, 480))

renderer = UIRenderer()
print("pil_ok:", renderer.pil_ok)

glasses_names = ["aviator", "round", "square", "wayfarer"]

# 1) Locked, high confidence, mic on, glasses recommended
state = build_ui_state(
    gender='male', voice_status='on', fps=28.7,
    face_shape='Oval', face_shape_confidence=82.3,
    analyzing=False, analysis_progress=(30, 30), near_edge=False, cnn_busy=False,
    skin_tone='Medium', skin_swatch_rgb=(178, 140, 110),
    glasses_idx=0, glasses_names=glasses_names, suggested=[0, 1, 2, 3], total_glasses=4,
)
out1 = renderer.render(frame, state)
cv2.imwrite(str(DEBUG_DIR) + '/ui_preview.jpg', out1)
print("saved data/debug/ui_preview.jpg", out1.shape)

# 2) Analyzing, near edge, cnn busy, mic off, glasses not recommended, low confidence colors unaffected
state2 = build_ui_state(
    gender='female', voice_status='off', fps=24.1,
    face_shape=None, face_shape_confidence=None,
    analyzing=True, analysis_progress=(14, 30), near_edge=True, cnn_busy=True,
    skin_tone='Very Light', skin_swatch_rgb=(220, 195, 180),
    glasses_idx=2, glasses_names=glasses_names, suggested=[0, 3], total_glasses=4,
)
out2 = renderer.render(frame, state2)
cv2.imwrite(str(DEBUG_DIR) + '/ui_preview_analyzing.jpg', out2)
print("saved data/debug/ui_preview_analyzing.jpg", out2.shape)

# 3) Locked, low confidence (red bar), deep skin tone
state3 = build_ui_state(
    gender='male', voice_status='listening', fps=29.9,
    face_shape='Square', face_shape_confidence=34.5,
    analyzing=False, analysis_progress=(30, 30), near_edge=False, cnn_busy=False,
    skin_tone='Deep', skin_swatch_rgb=(60, 45, 35),
    glasses_idx=3, glasses_names=glasses_names, suggested=[1, 3], total_glasses=4,
)
out3 = renderer.render(frame, state3)
cv2.imwrite(str(DEBUG_DIR) + '/ui_preview_lowconf.jpg', out3)
print("saved data/debug/ui_preview_lowconf.jpg", out3.shape)

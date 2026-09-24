"""
One-off diagnostic: runs the real MediaPipe FaceLandmarker on a handful
of different photos and prints exactly what compute_face_ratios/
classify_face_shape (B) and sample_skin_color/classify_skin_tone (C)
produce, so we have real printed evidence instead of guessing from
reading the code.
"""
import cv2
import mediapipe as mp
import glob

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # project root
from app.paths import DATASET_DIR
from app.face_tracking import MODEL_PATH, download_model
from app.face_shape_model import compute_face_ratios, classify_face_shape
from app.skin_tone import correct_lighting, sample_skin_color, classify_skin_tone

download_model()

BaseOptions           = mp.tasks.BaseOptions
FaceLandmarker        = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
)

paths = []
for shape in ['Heart', 'Oblong', 'Oval', 'Round', 'Square']:
    matches = glob.glob(str(DATASET_DIR / 'testing_set' / shape / '*'))[:2]
    paths.extend((p, shape) for p in matches)

with FaceLandmarker.create_from_options(options) as landmarker:
    for path, true_shape in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        corrected = correct_lighting(img)
        rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        if not result.face_landmarks:
            print(f"{path}: no face detected")
            continue
        landmarks = result.face_landmarks[0]

        ratios = compute_face_ratios(landmarks, w, h)
        predicted = classify_face_shape(landmarks, w, h)
        print(f"[B] {path} (true={true_shape}): "
              f"hr={ratios['hr']:.3f} fr={ratios['fr']:.3f} jr={ratios['jr']:.3f} "
              f"-> predicted={predicted}")

        sampled, low_conf = sample_skin_color(corrected, landmarks, w, h)
        if sampled is not None:
            tone = classify_skin_tone(sampled)
            print(f"[C] {path}: sampled_bgr={tuple(round(float(c),1) for c in sampled)} "
                  f"-> tone={tone} low_confidence={low_conf}")
        else:
            print(f"[C] {path}: sample_skin_color returned None")
        print()

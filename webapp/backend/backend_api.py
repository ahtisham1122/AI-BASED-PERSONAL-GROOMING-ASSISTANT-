"""
Flask backend for webapp/frontend/: a real POST /api/v1/analyze endpoint
replacing script.js's old Demo Mode mock data.

Reuses the exact same detection/classification/recommendation functions
main.py uses (face_shape_model.py, skin_tone.py, recommendations.py,
face_tracking.py) -- nothing here reimplements that logic, it's all one
image at a time instead of a live video feed.
"""
import os
import sys

# face_shape_model.py etc. live two folders up (webapp/backend/ -> webapp/
# -> project root), and load their model files (class_names.json,
# face_shape_cnn.tflite, ...) via bare relative paths at import time, so
# both the import path and the working directory need to point at the
# project root before importing them -- regardless of which directory
# this script is actually launched from.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_cors import CORS

from face_tracking import MODEL_PATH, download_model, _bbox_area
from face_shape_model import (
    MODE as FACE_SHAPE_MODE, CLASS_NAMES, classify_face_shape,
    predict_shape_probs, is_frontal_face, TIE_BREAK_MARGIN,
)
from skin_tone import correct_lighting, sample_skin_color, classify_skin_tone
from recommendations import (
    get_hair_rec, get_grooming_rec, get_glasses_rec,
    get_color_rec, get_avoid_colors, get_color_swatch,
)

app = Flask(__name__)
CORS(app)


# ──────────────────────────────────────────
# MediaPipe setup (IMAGE mode -- one photo per request, not a video
# stream), same landmarker config and "pick the largest face" logic
# as main.py's live loop (face_tracking.detect_landmarks).
# ──────────────────────────────────────────
_landmarker = None


def get_landmarker():
    global _landmarker
    if _landmarker is None:
        download_model()
        BaseOptions           = mp.tasks.BaseOptions
        FaceLandmarker        = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode     = mp.tasks.vision.RunningMode
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=3,  # detect up to 3 so we can pick the largest, same as main.py
        )
        _landmarker = FaceLandmarker.create_from_options(options)
    return _landmarker


def detect_face(landmarker, rgb_frame):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None, False
    if len(result.face_landmarks) == 1:
        return result.face_landmarks[0], False
    return max(result.face_landmarks, key=_bbox_area), True


def classify_face_shape_with_tiebreak(corrected_bgr, landmarks, w, h):
    """
    Same CNN + rule-based tie-break rule as face_shape_model's live
    CNNFaceShapeClassifier._freeze(), applied to this one prediction
    instead of an average over many video frames (there's only one
    frame here). predict_shape_probs() does the face-detector crop
    with the training's 25% margin internally.
    Returns (face_shape, confidence_percent_or_None).
    """
    if FACE_SHAPE_MODE == 'rule-based':
        return classify_face_shape(landmarks, w, h), None

    probs, _ = predict_shape_probs(corrected_bgr, w, h)
    if probs is None:
        return classify_face_shape(landmarks, w, h), None

    order = np.argsort(probs)[::-1]
    top_label,    top_conf    = CLASS_NAMES[order[0]], float(probs[order[0]])
    second_label, second_conf = CLASS_NAMES[order[1]], float(probs[order[1]])

    final_label = top_label
    if (top_conf - second_conf) < TIE_BREAK_MARGIN:
        rule_based = classify_face_shape(landmarks, w, h)
        if rule_based in (top_label, second_label):
            final_label = rule_based

    confidence = float(probs[CLASS_NAMES.index(final_label)]) * 100
    return final_label, confidence


def normalize_gender(raw):
    """Frontend sends 'men'/'women'; recommendations.py expects 'male'/'female'."""
    return 'female' if (raw or '').strip().lower() in ('women', 'female', 'f') else 'male'


def to_hex(rgb):
    r, g, b = rgb
    return '#{:02x}{:02x}{:02x}'.format(int(r), int(g), int(b))


def color_list(csv_names):
    """Turns a "Navy, Forest Green, ..." string from recommendations.py into [{name, hex}, ...]."""
    if csv_names == "---":
        return []
    return [
        {"name": name.strip(), "hex": to_hex(get_color_swatch(name.strip()))}
        for name in csv_names.split(",")
    ]


# Only 4 glasses SVGs exist as real assets (assets/glasses/); recommendations.py's
# frame-TYPE names are broader than that, so each one maps to whichever of the 4
# actual assets looks closest. Anything not listed falls back to rectangle.svg
# (the frontend's <img onerror> also falls back to the same file as a second
# safety net if an asset ever fails to load).
GLASSES_IMAGE_MAP = {
    "Rectangle frames":     "rectangle.svg",
    "Square frames":        "rectangle.svg",
    "Round frames":         "round.svg",
    "Oval frames":          "round.svg",
    "Thin/rimless frames":  "round.svg",
    "Rimless frames":       "round.svg",
    "Aviator":              "aviator.svg",
    "Oversized frames":     "aviator.svg",
    "Browline frames":      "wayfarer.svg",
}


def glasses_image(name):
    return f"assets/glasses/{GLASSES_IMAGE_MAP.get(name, 'rectangle.svg')}"


@app.route('/api/v1/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "missing_image", "message": "No image file was uploaded."}), 400

    gender = normalize_gender(request.form.get('gender'))
    # The frontend also sends "occasion" (casual/formal/party) -- Flask
    # accepts it as a form field fine without us reading it, but nothing
    # here uses it: recommendations.py has no occasion-based tables.

    file_bytes = np.frombuffer(request.files['image'].read(), dtype=np.uint8)
    frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return jsonify({"error": "bad_image", "message": "Couldn't read that image file."}), 400

    corrected = correct_lighting(frame_bgr)
    h, w = corrected.shape[:2]
    corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)

    landmarks, multiple_faces = detect_face(get_landmarker(), corrected_rgb)
    if landmarks is None:
        return jsonify({
            "error": "no_face",
            "message": "No face detected in this photo. Try a clearer, front-facing photo with good lighting.",
        }), 422

    face_shape, confidence = classify_face_shape_with_tiebreak(corrected, landmarks, w, h)

    skin_tone = None
    skin_swatch_hex = None
    skin_tone_low_confidence = False
    sampled, low_conf = sample_skin_color(corrected, landmarks, w, h)
    if sampled is not None:
        skin_tone = classify_skin_tone(sampled)
        skin_swatch_hex = to_hex((sampled[2], sampled[1], sampled[0]))  # BGR -> RGB
        skin_tone_low_confidence = low_conf

    glasses = [
        {"name": rec["name"], "image": glasses_image(rec["name"]), "reason": rec["why"]}
        for rec in get_glasses_rec(face_shape)
    ]

    return jsonify({
        "faceShape": face_shape,
        "confidence": confidence,
        "skinTone": skin_tone,
        "skinToneSwatch": skin_swatch_hex,
        "skinToneLowConfidence": skin_tone_low_confidence,
        "multipleFaces": multiple_faces,
        "frontal": is_frontal_face(landmarks, w, h),
        "glasses": glasses,
        "hairstyle": get_hair_rec(gender, face_shape),
        "grooming": get_grooming_rec(gender, face_shape),
        "outfitColors": {
            "wear": color_list(get_color_rec(gender, skin_tone)),
            "avoid": color_list(get_avoid_colors(skin_tone)),
        },
    })


if __name__ == '__main__':
    print(f"[backend_api] Face shape model mode: {FACE_SHAPE_MODE}")
    # use_reloader=False: the reloader re-launches the script using a path
    # relative to the process's cwd, which we deliberately chdir() above --
    # that combination makes the reloader relaunch a nonexistent path.
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)

"""
Flask backend for webapp/frontend/: a real POST /api/v1/analyze endpoint
replacing script.js's old Demo Mode mock data.

Reuses the exact same detection/classification/recommendation functions
main.py uses (face_shape_model.py, skin_tone.py, recommendations.py,
face_tracking.py) -- nothing here reimplements that logic, it's all one
image at a time instead of a live video feed.
"""
import json
import os
import sys
from functools import lru_cache

# The app/ package lives two folders up (webapp/backend/ -> webapp/ ->
# project root). Its modules build every model/asset path from their own
# location (app/paths.py), so only the import path needs setting here.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify, Response, abort
from flask_cors import CORS

from app.face_tracking import MODEL_PATH, download_model, _bbox_area
from app.face_shape_model import (
    MODE as FACE_SHAPE_MODE, CLASS_NAMES, classify_face_shape,
    predict_shape_probs, is_frontal_face, TIE_BREAK_MARGIN,
)
from app.paths import ASSETS_DIR
from app.ar_overlay import _auto_crop
from app.skin_tone import correct_lighting, sample_skin_color, classify_skin_tone
from app.recommendations import (
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


# Glasses images: served straight from the SAME folder the desktop app's
# AR try-on loads (assets/glasses/processed/), so a PNG added there shows
# up on both. glasses_styles.json tags each PNG with frame types so a
# recommendation like "Round frames" can point at every matching PNG.
GLASSES_DIR = str(ASSETS_DIR / 'glasses' / 'processed')
GLASSES_STYLES_PATH = str(ASSETS_DIR / 'glasses' / 'glasses_styles.json')


def _load_glasses_styles():
    try:
        with open(GLASSES_STYLES_PATH, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def list_web_glasses():
    """Every PNG in GLASSES_DIR (re-read each call, so new files appear without a restart), minus hide_on_web ones."""
    styles = _load_glasses_styles()
    ids = sorted(
        (os.path.splitext(f)[0] for f in os.listdir(GLASSES_DIR) if f.lower().endswith('.png')),
        key=lambda s: (len(s), s),  # natural order: glasses2 before glasses10
    )
    return [
        {"id": gid, "styles": styles.get(gid, {}).get("styles", []), "image": f"{request.host_url}glasses/{gid}.png"}
        for gid in ids if not styles.get(gid, {}).get("hide_on_web")
    ]


def style_key(rec_name):
    """recommendations.py frame name -> glasses_styles.json tag, e.g. 'Round frames' -> 'round', 'Thin/rimless frames' -> 'thin'."""
    key = rec_name.lower().replace(" frames", "").strip()
    return {"thin/rimless": "thin", "rimless": "thin"}.get(key, key)


@lru_cache(maxsize=64)  # 64 > the ~13 frames we have, so every cropped PNG stays cached after its first request
def _cropped_png(path, mtime):
    """Trims the transparent margin (same _auto_crop the desktop app uses) so the web size slider behaves the same for every frame. mtime in the key = re-crop if the file changes."""
    rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
        return None
    ok, buf = cv2.imencode('.png', _auto_crop(rgba))
    return buf.tobytes() if ok else None


@app.route('/glasses/<gid>.png')
def glasses_png(gid):
    # Only names that really exist in the folder -- never build a path from raw URL text.
    if gid + '.png' not in os.listdir(GLASSES_DIR) or _load_glasses_styles().get(gid, {}).get("hide_on_web"):
        abort(404)
    path = os.path.join(GLASSES_DIR, gid + '.png')
    data = _cropped_png(path, os.path.getmtime(path))
    if data is None:
        abort(404)
    return Response(data, mimetype='image/png')


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

    all_glasses = list_web_glasses()
    glasses = []
    for rec in get_glasses_rec(face_shape):
        matches = [g for g in all_glasses if style_key(rec["name"]) in g["styles"]]
        glasses.append({
            "name": rec["name"],
            "reason": rec["why"],
            "ids": [g["id"] for g in matches],
            "images": [g["image"] for g in matches],
            # Card thumbnail; falls back to the first frame if no PNG is tagged with this type yet.
            "image": (matches or all_glasses or [{"image": None}])[0]["image"],
        })

    return jsonify({
        "faceShape": face_shape,
        "confidence": confidence,
        "skinTone": skin_tone,
        "skinToneSwatch": skin_swatch_hex,
        "skinToneLowConfidence": skin_tone_low_confidence,
        "multipleFaces": multiple_faces,
        "frontal": is_frontal_face(landmarks, w, h),
        "glasses": glasses,
        "allGlasses": all_glasses,   # every web-visible frame, for the try-on Next/Previous buttons
        "hairstyle": get_hair_rec(gender, face_shape),
        "grooming": get_grooming_rec(gender, face_shape),
        "outfitColors": {
            "wear": color_list(get_color_rec(gender, skin_tone)),
            "avoid": color_list(get_avoid_colors(skin_tone)),
        },
    })


if __name__ == '__main__':
    print(f"[backend_api] Face shape model mode: {FACE_SHAPE_MODE}")
    # use_reloader=False: the reloader would load the models a second time
    # in a child process for no benefit here.
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)

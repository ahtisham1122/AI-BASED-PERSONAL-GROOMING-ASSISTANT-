"""
Streamlit web version of the AI Grooming Assistant.

Single-image version of the same app: upload a photo or take one with
the browser camera, and get the same face shape / skin tone /
recommendation results main.py produces live — using the exact same
detection and recommendation functions, just called once per image
instead of continuously on video. Kept fully separate from main.py:
this file does not import main.py or ui.py (the desktop UI, which is
built around cv2/PIL and a running camera loop) and nothing here is
imported back into the desktop app either.
"""
import io
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

from face_tracking import MODEL_PATH, download_model, _bbox_area
from face_shape_model import (
    MODE as FACE_SHAPE_MODE, CLASS_NAMES, classify_face_shape,
    predict_shape_probs, is_frontal_face,
)
from skin_tone import correct_lighting, sample_skin_color, classify_skin_tone
from recommendations import (
    get_hair_rec, get_grooming_rec, get_glasses_rec,
    get_color_rec, get_avoid_colors, get_color_swatch,
)

# Same palette as ui.py (the desktop app's PIL-rendered panel), so the
# two apps read as the same product even though this one is plain
# HTML/CSS instead of PIL.
BG_CARD    = "#1E2228"
CARD_BORDER = "#2A2F37"
ACCENT     = "#4AADB0"
TEXT_WHITE = "#E8EAED"
TEXT_DIM   = "#8E949D"
GREEN      = "#56C489"
AMBER      = "#E6AC3C"
RED        = "#DE5A5A"

def _inject_style():
    st.markdown(f"""
<style>
.gc-card {{
    background-color: {BG_CARD};
    border: 1px solid {CARD_BORDER};
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
}}
.gc-heading {{
    color: {TEXT_DIM};
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 10px;
}}
.gc-value {{
    color: {TEXT_WHITE};
    font-size: 26px;
    font-weight: 700;
}}
.gc-item {{
    color: {TEXT_WHITE};
    font-size: 15px;
    font-weight: 600;
    margin-top: 10px;
}}
.gc-sub {{
    color: {TEXT_DIM};
    font-size: 13px;
    margin-top: 2px;
}}
.gc-bar-bg {{
    background-color: #34383F;
    border-radius: 6px;
    height: 10px;
    width: 100%;
    margin-top: 10px;
}}
.gc-bar-fill {{
    height: 10px;
    border-radius: 6px;
}}
.gc-swatch {{
    display: inline-block;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    border: 2px solid #444;
    vertical-align: middle;
    margin-right: 12px;
}}
.gc-swatch-small {{
    display: inline-block;
    width: 13px;
    height: 13px;
    border-radius: 50%;
    vertical-align: middle;
    margin-right: 6px;
    border: 1px solid #555;
}}
.gc-pill {{
    display: inline-block;
    background-color: #2A2F37;
    color: {TEXT_WHITE};
    padding: 4px 12px 4px 8px;
    border-radius: 12px;
    font-size: 13px;
    margin: 3px 6px 3px 0;
}}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────
# MediaPipe setup (IMAGE mode, not the desktop's VIDEO mode — there's
# no continuous stream/timestamp to track for a single uploaded photo)
# ──────────────────────────────────────────
@st.cache_resource(show_spinner="Loading face landmark model...")
def get_landmarker():
    download_model()
    BaseOptions           = mp.tasks.BaseOptions
    FaceLandmarker        = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=3,  # detect up to 3 so we can pick the largest, same as the desktop app
    )
    return FaceLandmarker.create_from_options(options)


def detect_face(landmarker, rgb_frame):
    """Same 'pick the largest face' logic as face_tracking.detect_landmarks, adapted for a single static image."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None, False
    if len(result.face_landmarks) == 1:
        return result.face_landmarks[0], False
    return max(result.face_landmarks, key=_bbox_area), True


@st.cache_data(show_spinner="Analyzing photo...")
def analyze_image(image_bytes):
    """
    Runs the same detection/classification pipeline main.py runs per
    frame, once, on one image. Returns a plain dict of primitives
    (not the raw MediaPipe landmarks) so Streamlit can cache it keyed
    on the image bytes — re-uploading the same photo, or just toggling
    the gender selector, won't re-run detection.
    """
    landmarker = get_landmarker()

    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    frame_rgb = np.array(pil_image)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    corrected = correct_lighting(frame_bgr)
    corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
    landmarks, multiple_faces = detect_face(landmarker, corrected_rgb)

    if landmarks is None:
        return {'face_found': False, 'multiple_faces': False}

    h, w = corrected.shape[:2]
    frontal = is_frontal_face(landmarks, w, h)

    # Face shape: same CNN-with-rule-based-fallback logic as main.py,
    # just one inference instead of averaging over many video frames
    # (the desktop app's multi-frame averaging exists to smooth out
    # frame-to-frame noise in a live feed, which doesn't apply to a
    # single photo).
    face_shape = None
    confidence = None
    if FACE_SHAPE_MODE != 'rule-based':
        probs, _ = predict_shape_probs(corrected, w, h)
        if probs is not None:
            idx = int(np.argmax(probs))
            face_shape = CLASS_NAMES[idx]
            confidence = float(probs[idx]) * 100
        else:
            face_shape = classify_face_shape(landmarks, w, h)
    else:
        face_shape = classify_face_shape(landmarks, w, h)

    skin_tone = None
    skin_swatch_rgb = None
    skin_tone_low_confidence = False
    sampled, low_conf = sample_skin_color(corrected, landmarks, w, h)
    if sampled is not None:
        skin_tone = classify_skin_tone(sampled)
        skin_swatch_rgb = (int(sampled[2]), int(sampled[1]), int(sampled[0]))  # BGR -> RGB
        skin_tone_low_confidence = low_conf

    return {
        'face_found': True,
        'multiple_faces': multiple_faces,
        'frontal': frontal,
        'face_shape': face_shape,
        'confidence': confidence,
        'skin_tone': skin_tone,
        'skin_swatch_rgb': skin_swatch_rgb,
        'skin_tone_low_confidence': skin_tone_low_confidence,
    }


# ──────────────────────────────────────────
# Small HTML helpers (native Streamlit widgets can't do arbitrary
# colors for bars/swatches, so these render the same visual language
# as ui.py's PIL cards using plain HTML/CSS instead)
# ──────────────────────────────────────────
def confidence_color(pct):
    if pct >= 60:
        return GREEN
    elif pct >= 40:
        return AMBER
    return RED


def card_open(title):
    st.markdown(f'<div class="gc-card"><div class="gc-heading">{title}</div>', unsafe_allow_html=True)


def card_close():
    st.markdown('</div>', unsafe_allow_html=True)


def color_pills(names):
    html = ""
    for name in names:
        r, g, b = get_color_swatch(name)
        html += (f'<span class="gc-pill">'
                 f'<span class="gc-swatch-small" style="background-color: rgb({r},{g},{b});"></span>'
                 f'{name}</span>')
    st.markdown(html, unsafe_allow_html=True)


# ──────────────────────────────────────────
# Page
# ──────────────────────────────────────────
def main():
    st.set_page_config(page_title="AI Grooming Assistant", layout="wide")
    _inject_style()

    st.title("AI Grooming Assistant")
    st.caption("Upload a photo or use your camera for face shape, skin tone, and style recommendations.")

    with st.sidebar:
        st.header("Settings")
        gender_label = st.radio("Gender", ["Male", "Female"])
        gender = gender_label.lower()

        st.divider()
        source = st.radio("Image source", ["Upload a photo", "Use camera"])

        if FACE_SHAPE_MODE == 'rule-based':
            st.warning("Running in rule-based mode — the trained CNN model files weren't found.")

    col_photo, col_results = st.columns([1, 1])

    image_bytes = None
    with col_photo:
        if source == "Upload a photo":
            uploaded = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                image_bytes = uploaded.getvalue()
        else:
            captured = st.camera_input("Take a photo")
            if captured is not None:
                image_bytes = captured.getvalue()

        if image_bytes is not None:
            st.image(image_bytes, caption="Analyzed photo", use_container_width=True)

    if image_bytes is None:
        st.info("Upload a photo or take one with your camera to get started.")
        st.stop()

    result = analyze_image(image_bytes)

    if not result['face_found']:
        st.error("No face detected in this photo. Try a clearer, more front-facing photo with good lighting.")
        st.stop()

    with col_results:
        if result['multiple_faces']:
            st.warning("Multiple faces detected — analyzing the largest one.")
        if not result['frontal']:
            st.warning("This photo isn't fully front-facing / the face is small or near the edge — "
                       "results may be less accurate. A straight-on, well-lit photo works best.")

        # ── Face shape ──
        card_open("Face Shape")
        st.markdown(f'<div class="gc-value">{result["face_shape"] or "Unknown"}</div>', unsafe_allow_html=True)
        conf = result['confidence']
        if conf is not None:
            color = confidence_color(conf)
            st.markdown(f"""
                <div class="gc-bar-bg"><div class="gc-bar-fill" style="width: {conf:.0f}%; background-color: {color};"></div></div>
                <div class="gc-sub">{conf:.0f}% confidence</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="gc-sub">Rule-based estimate (no confidence score)</div>', unsafe_allow_html=True)
        card_close()

        # ── Skin tone ──
        card_open("Skin Tone")
        if result['skin_tone']:
            r, g, b = result['skin_swatch_rgb']
            st.markdown(
                f'<span class="gc-swatch" style="background-color: rgb({r},{g},{b});"></span>'
                f'<span class="gc-value" style="font-size:20px;">{result["skin_tone"]}</span>',
                unsafe_allow_html=True,
            )
            if result['skin_tone_low_confidence']:
                st.markdown(f'<div class="gc-sub" style="color:{AMBER};margin-top:6px;">'
                            f'Rough estimate — part of the face looks shadowed</div>', unsafe_allow_html=True)
            st.markdown('<div class="gc-sub" style="margin-top:6px;">'
                        'For best results, remove hats and ensure even lighting</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="gc-sub">Couldn\'t reliably sample skin tone from this photo.</div>',
                        unsafe_allow_html=True)
        card_close()

        face_shape = result['face_shape']
        skin_tone  = result['skin_tone']

        # ── Hairstyle ──
        card_open("Hairstyle")
        hair_recs = get_hair_rec(gender, face_shape)
        if hair_recs:
            for rec in hair_recs:
                tag = ' <span style="color:{};font-size:12px;">less popular</span>'.format(AMBER) \
                    if rec.get('less_popular') else ''
                st.markdown(f'<div class="gc-item">{rec["name"]}{tag}</div>'
                            f'<div class="gc-sub">{rec["why"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="gc-sub">---</div>', unsafe_allow_html=True)
        card_close()

        # ── Grooming ──
        card_open("Grooming")
        tips = get_grooming_rec(gender, face_shape)
        if tips:
            for tip in tips:
                st.markdown(f'<div class="gc-sub">&bull; {tip}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="gc-sub">---</div>', unsafe_allow_html=True)
        card_close()

        # ── Outfit colors ──
        card_open("Outfit Colors")
        color_rec = get_color_rec(gender, skin_tone)
        if color_rec != "---":
            color_pills([c.strip() for c in color_rec.split(",")])
            avoid = get_avoid_colors(skin_tone)
            if avoid != "---":
                st.markdown('<div class="gc-sub" style="margin-top:8px;">Avoid:</div>', unsafe_allow_html=True)
                color_pills([c.strip() for c in avoid.split(",")])
        else:
            st.markdown('<div class="gc-sub">Detect skin tone first.</div>', unsafe_allow_html=True)
        card_close()

        # ── Glasses ──
        card_open("Glasses")
        glasses_recs = get_glasses_rec(face_shape)
        if glasses_recs:
            for rec in glasses_recs:
                tag = ' <span style="color:{};font-size:12px;">less popular</span>'.format(AMBER) \
                    if rec.get('less_popular') else ''
                st.markdown(f'<div class="gc-item">{rec["name"]}{tag}</div>'
                            f'<div class="gc-sub">{rec["why"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="gc-sub">---</div>', unsafe_allow_html=True)
        card_close()


if __name__ == "__main__":
    main()

"""
Face shape classification: a CNN (trained in Colab on face crops) when
available, with a geometric ratio-based rule classifier as a fallback
if the trained model files are missing or fail to load.

The CNN itself runs on a background thread (CNNFaceShapeClassifier)
so the camera loop never blocks waiting on it.
"""
import cv2
import numpy as np
import json
import os
import random
import glob
import threading
import time
import urllib.request
import mediapipe as mp

# ──────────────────────────────────────────
# Rule-based fallback (geometric ratios)
# ──────────────────────────────────────────
def dist(a, b, w, h):
    """
    Euclidean distance between two MediaPipe normalized landmarks, in
    PIXELS. MediaPipe's landmark.x is normalized by image width and
    .y by image height separately — on a non-square frame those are
    two different units, so computing sqrt(dx^2+dy^2) directly on the
    raw normalized values silently distorts vertical-vs-horizontal
    measurements (confirmed by testing: on a 640x480 landscape frame
    it inflates height-based ratios, on a portrait photo it deflates
    them — this was the actual cause of "Oblong for everyone" and
    "everyone is Round" on different image shapes). Scaling by (w, h)
    first puts both axes in the same real-world unit before measuring.
    """
    return np.sqrt(
        ((a.x - b.x) * w) ** 2 +
        ((a.y - b.y) * h) ** 2
    )


def compute_face_ratios(landmarks, w, h):
    """
    Computes geometric ratios from MediaPipe face landmarks. Used both
    by the rule-based classifier below and by build_dataset_from_kaggle.py
    to build ML training data — keeping the math in one place means the
    app and the trained model always measure faces the same way.
    Returns None if the face is degenerate (zero width) so callers can
    skip it instead of dividing by zero. Needs the frame's width/height
    to convert normalized landmark coordinates to real pixel distances.
    """
    fh  = dist(landmarks[10],  landmarks[152], w, h)  # forehead top to chin bottom = face height
    cw  = dist(landmarks[234], landmarks[454], w, h)  # cheek to cheek = cheekbone width
    fw  = dist(landmarks[70],  landmarks[300], w, h)  # forehead width
    jw  = dist(landmarks[58],  landmarks[288], w, h)  # jaw width
    tw  = dist(landmarks[127], landmarks[356], w, h)  # temple width, near top of ears
    chw = dist(landmarks[172], landmarks[397], w, h)  # chin width, narrow point at bottom of chin

    if cw == 0:
        return None

    # Face length-to-width ratio: height over whichever width we
    # measured is widest. The cheekbones aren't always the widest
    # point (e.g. a square face can be widest at the jaw).
    widest = max(cw, fw, jw, tw)
    flr = fh / widest

    # Jaw angle: angle at the chin (landmark 152) between the lines to
    # the two jaw corners (58, 288). A small angle = pointed chin
    # (Heart/Oval), a large angle = flat, rounded jawline (Round/Square).
    # Uses pixel-scaled vectors too, for the same aspect-ratio reason.
    v1 = np.array([(landmarks[58].x  - landmarks[152].x) * w, (landmarks[58].y  - landmarks[152].y) * h])
    v2 = np.array([(landmarks[288].x - landmarks[152].x) * w, (landmarks[288].y - landmarks[152].y) * h])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # guard float rounding past [-1, 1]
    jaw_angle = np.degrees(np.arccos(cos_angle))

    return {
        'hr': fh / cw,
        'fr': fw / cw,
        'jr': jw / cw,
        'flr': flr,
        'jaw_angle': jaw_angle,
        'temple_ratio': tw / cw,
        'chin_ratio': chw / cw,
    }


def classify_face_shape(landmarks, w, h):
    """Rule-based fallback, used only when the CNN model can't be loaded."""
    ratios = compute_face_ratios(landmarks, w, h)
    if ratios is None:
        return None

    hr = ratios['hr']
    fr = ratios['fr']
    jr = ratios['jr']
    print(f"[face_shape_model] rule-based ratios: hr={hr:.3f} fr={fr:.3f} jr={jr:.3f}")

    if hr > 1.55:
        return "Oblong"
    elif hr > 1.30:
        if fr > 0.88 and jr > 0.82:
            return "Square"
        elif fr > jr + 0.12:
            return "Heart"
        else:
            return "Oval"
    elif hr > 1.10:
        if jr > 0.88 and fr > 0.88:
            return "Square"
        elif fr > jr + 0.12:
            return "Heart"
        else:
            return "Oval"
    else:
        return "Round"


# ──────────────────────────────────────────
# CNN model loading (tflite -> keras -> rule-based)
# ──────────────────────────────────────────
CLASS_NAMES_PATH = 'class_names.json'
TFLITE_PATH      = 'face_shape_cnn.tflite'
KERAS_PATH       = 'face_shape_cnn.keras'
IMG_SIZE         = 224
MARGIN           = 0.25  # extra margin around the face box, matches the training crop

CLASS_NAMES   = None
_interpreter  = None
_input_index  = None
_output_index = None
_keras_model  = None
MODE          = None  # 'tflite', 'keras', or 'rule-based'
_last_inference_ms = 0.0  # set by predict_shape_probs(), read by get_last_inference_ms()

try:
    with open(CLASS_NAMES_PATH) as f:
        CLASS_NAMES = json.load(f)
except Exception as e:
    print(f"[face_shape_model] Couldn't load {CLASS_NAMES_PATH}: {e}")

if CLASS_NAMES is not None:
    try:
        import tensorflow as tf
        _interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
        _interpreter.allocate_tensors()
        _input_index  = _interpreter.get_input_details()[0]['index']
        _output_index = _interpreter.get_output_details()[0]['index']
        MODE = 'tflite'
    except Exception as e:
        print(f"[face_shape_model] Couldn't load {TFLITE_PATH} ({e}), trying Keras model...")
        _interpreter = None

    if MODE is None:
        try:
            import tensorflow as tf
            _keras_model = tf.keras.models.load_model(KERAS_PATH)
            MODE = 'keras'
        except Exception as e:
            print(f"[face_shape_model] Couldn't load {KERAS_PATH} ({e}), "
                  f"falling back to the rule-based classifier.")

if MODE is None:
    MODE = 'rule-based'

print(f"[face_shape_model] Face shape model mode: {MODE}")


def get_face_bbox(landmarks, w, h):
    """
    Tight bounding box around all face landmarks (NOT expanded/square)
    — used only as a cheap, landmark-based approximation of "where is
    the face and how big is it", for the frontal/size/edge/identity
    checks below. The real CNN crop uses detect_face_box() instead,
    which runs an actual face detector matching how the training data
    was built (see MARGIN / _expand_to_square).
    """
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    return min(xs), min(ys), max(xs), max(ys)


# How far the nose tip may drift from the midpoint between the eyes,
# as a fraction of the eye-to-eye distance, before we call the head
# "turned" rather than frontal. 0.15 was picked by eye — small enough
# to reject an obviously turned head, loose enough to allow natural
# minor head wobble.
NOSE_OFFSET_RATIO_MAX = 0.15
# Faces narrower than this (in pixels, at the app's 640x480 capture
# size) are too small/far away to crop cleanly for the CNN.
MIN_FACE_BOX_WIDTH = 150
# If the (unexpanded) face box comes within this many pixels of the
# frame edge, the camera itself may be clipping part of the face —
# no amount of crop padding can recover detail that was never
# captured, so these frames are skipped rather than fed to the CNN.
EDGE_MARGIN_PX = 20


def is_frontal_face(landmarks, w, h):
    """
    True if the head is roughly facing the camera (nose centered
    between the eyes) and close enough that the face box is a decent
    size. Used to gate which frames are worth sending to the CNN —
    a turned or tiny face produces a crop that doesn't look like the
    training data, which was a source of flickery predictions.
    """
    nose      = landmarks[4]
    left_eye  = landmarks[33]
    right_eye = landmarks[263]

    eye_span = abs(right_eye.x - left_eye.x)
    if eye_span == 0:
        return False
    eye_cx = (left_eye.x + right_eye.x) / 2
    offset_ratio = abs(nose.x - eye_cx) / eye_span
    if offset_ratio > NOSE_OFFSET_RATIO_MAX:
        return False

    x1, y1, x2, y2 = get_face_bbox(landmarks, w, h)
    return (x2 - x1) >= MIN_FACE_BOX_WIDTH


def is_face_near_edge(landmarks, w, h):
    """
    True if the face's own bounding box (before any margin expansion)
    is close enough to a frame edge that the camera may be clipping
    part of it. Distinct from padding: padding fixes the *expanded*
    square box going out of bounds, but can't invent detail the
    camera never captured in the first place.
    """
    x1, y1, x2, y2 = get_face_bbox(landmarks, w, h)
    return (x1 <= EDGE_MARGIN_PX or y1 <= EDGE_MARGIN_PX or
            x2 >= w - EDGE_MARGIN_PX or y2 >= h - EDGE_MARGIN_PX)


# ──────────────────────────────────────────
# Face detector (BlazeFace) for CNN cropping
# ──────────────────────────────────────────
# Training crops came from a face DETECTOR box (BlazeFace short range)
# expanded 25% into a square, not from the landmark min/max box — the
# landmark box is zoomed in and shifted down relative to that (confirmed
# visually via debug crops: hairline cut off, neck/collar visible at
# the bottom). Using the actual detector here matches the training
# framing instead of approximating it.
FACE_DETECTOR_PATH = 'blaze_face_short_range.tflite'
FACE_DETECTOR_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'face_detector/blaze_face_short_range/float16/1/'
    'blaze_face_short_range.tflite'
)

_face_detector = None


def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        if not os.path.exists(FACE_DETECTOR_PATH):
            print("[face_shape_model] Downloading face detector model...")
            urllib.request.urlretrieve(FACE_DETECTOR_URL, FACE_DETECTOR_PATH)
            print("[face_shape_model] Face detector model ready!")
        BaseOptions         = mp.tasks.BaseOptions
        FaceDetector        = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=FACE_DETECTOR_PATH),
        )
        _face_detector = FaceDetector.create_from_options(options)
    return _face_detector


def detect_face_box(frame_bgr):
    """
    Runs the BlazeFace detector on frame_bgr. Returns its raw
    detection box in pixels (x1, y1, x2, y2), or None if no face was
    found. This is the same *kind* of box the training crops came
    from — tighter and differently framed than the landmark min/max
    box.
    """
    detector = _get_face_detector()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    if not result.detections:
        return None
    box = result.detections[0].bounding_box
    return box.origin_x, box.origin_y, box.origin_x + box.width, box.origin_y + box.height


def _expand_to_square(x1, y1, x2, y2, margin):
    """Expands a box into a square with `margin` extra on each side. Deliberately NOT clamped to frame bounds — see _crop_square_padded."""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    half = side / 2
    return cx - half, cy - half, cx + half, cy + half


def _crop_square_padded(frame_bgr, x1, y1, x2, y2):
    """
    Crops (x1,y1)-(x2,y2) out of frame_bgr. If the box extends past
    the frame edges, pads with black borders (cv2.copyMakeBorder)
    instead of clamping. Clamping would turn the box non-square, and
    resizing a non-square crop to 224x224 stretches the face — that
    stretch was making faces look artificially Oblong.
    """
    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    h, w = frame_bgr.shape[:2]
    pad_left   = max(0, -x1)
    pad_top    = max(0, -y1)
    pad_right  = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        frame_bgr = cv2.copyMakeBorder(
            frame_bgr, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        x1, x2 = x1 + pad_left, x2 + pad_left
        y1, y2 = y1 + pad_top, y2 + pad_top

    return frame_bgr[y1:y2, x1:x2]


def _crop_and_resize(frame_bgr, w, h):
    """Shared by predict_shape_probs and save_debug_crop so they crop identically."""
    box = detect_face_box(frame_bgr)
    if box is None:
        return None
    x1, y1, x2, y2 = _expand_to_square(*box, MARGIN)
    crop = _crop_square_padded(frame_bgr, x1, y1, x2, y2)
    if crop.size == 0:
        return None
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))


def predict_shape_probs(frame_bgr, w, h):
    """
    Crops the face out of frame_bgr and runs it through the CNN.
    Returns (probs, resized_crop_bgr), or (None, None) if the CNN
    isn't available or the crop is unusable. probs is the raw softmax
    output, in CLASS_NAMES order.
    """
    if MODE == 'rule-based':
        return None, None

    resized = _crop_and_resize(frame_bgr, w, h)
    if resized is None:
        return None, None

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)  # cv2 gives BGR, model trained on RGB
    # Model has rescaling built in, so it takes raw 0-255 float32.
    batch = np.expand_dims(rgb.astype(np.float32), axis=0)
    print(f"[face_shape_model] input tensor: dtype={batch.dtype} shape={batch.shape} "
          f"min={batch.min():.1f} max={batch.max():.1f} mean={batch.mean():.1f}")

    # Timed separately from cropping/preprocessing above, so this is
    # just the model's own prediction time (for performance_logger.py).
    t0 = time.perf_counter()
    if MODE == 'tflite':
        _interpreter.set_tensor(_input_index, batch)
        _interpreter.invoke()
        probs = _interpreter.get_tensor(_output_index)[0]
    else:  # 'keras'
        probs = _keras_model.predict(batch, verbose=0)[0]
    global _last_inference_ms
    _last_inference_ms = (time.perf_counter() - t0) * 1000

    return probs, resized


def get_last_inference_ms():
    """Time (ms) the most recent CNN prediction took. 0.0 if none has run yet."""
    return _last_inference_ms


def _sample_training_crop():
    """Grabs one random dataset image and crops it the same way, for a framing comparison."""
    paths = glob.glob('face_shape_dataset/testing_set/*/*')
    if not paths:
        return None
    img = cv2.imread(random.choice(paths))
    if img is None:
        return None
    h, w = img.shape[:2]
    return _crop_and_resize(img, w, h)


def save_debug_crop(frame_bgr, landmarks, w, h, folder='debug'):
    """
    Saves the exact 224x224 crop the CNN would be fed right now,
    next to a same-size crop of a random training image (same crop
    function, side by side in one image) so framing can be compared
    directly. Returns the saved path, or None if the face doesn't
    currently qualify for a real CNN submission (same frontal/size
    gate as predict_shape_probs would apply).
    """
    if not is_frontal_face(landmarks, w, h):
        return None
    resized = _crop_and_resize(frame_bgr, w, h)
    if resized is None:
        return None

    training_crop = _sample_training_crop()
    if training_crop is not None:
        divider = np.full((IMG_SIZE, 4, 3), 255, dtype=np.uint8)
        resized = np.hstack([resized, divider, training_crop])
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"debug_crop_{int(time.time())}.jpg")
    cv2.imwrite(filename, resized)
    return filename


# How many frontal-face predictions to average before locking the
# result. At the app's submission rate (every 5th frame, ~6/sec on a
# 30fps camera) this takes roughly 3-5 seconds, matching the
# "Analyzing..." window shown in the UI.
ANALYSIS_FRAMES = 30
# If the face disappears for longer than this, forget the lock and
# start analyzing again once a face reappears — a brief look-away
# shouldn't reset it, but genuinely leaving should.
FACE_LOST_RESET_SECONDS = 2.0
# If the top two averaged class probabilities are within this margin
# of each other, the CNN is effectively unsure between them; let the
# geometric rule-based classifier cast the deciding vote if it agrees
# with either candidate.
TIE_BREAK_MARGIN = 0.10
# If the (landmark-based) face box center moves by more than this
# fraction of the frame size, or its size changes by more than this
# fraction, between one submitted frame and the next, treat it as a
# different person sitting down rather than the same person moving —
# and restart analysis from scratch instead of blending both people's
# predictions into one average.
IDENTITY_CENTER_JUMP_RATIO = 0.15
IDENTITY_SIZE_JUMP_RATIO   = 0.40


def _box_center_size(landmarks, w, h):
    x1, y1, x2, y2 = get_face_bbox(landmarks, w, h)
    return (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)


def _identity_jumped(cx, cy, size, ref_cx, ref_cy, ref_size, w, h):
    if ref_size == 0:
        return False
    center_shift = np.hypot((cx - ref_cx) / w, (cy - ref_cy) / h)
    size_change  = abs(size - ref_size) / ref_size
    return (center_shift > IDENTITY_CENTER_JUMP_RATIO or
            size_change > IDENTITY_SIZE_JUMP_RATIO)


class CNNFaceShapeClassifier:
    """
    Runs CNN face shape prediction on a background thread so the
    camera loop never waits on it, with an "analysis lock": rather
    than re-predicting every frame (which flickered between classes),
    it collects ANALYSIS_FRAMES frontal-face predictions, averages
    their softmax probability vectors, and freezes on that averaged
    result. Call restart() to re-run analysis (e.g. on a keypress),
    and note_face_seen() every frame so a long-missing face
    auto-resets the lock next time one appears.
    """
    def __init__(self):
        self._lock    = threading.Lock()
        self._pending = None
        self._busy    = False
        self._stop    = False

        self._prob_history  = []
        self._locked        = False
        self._label         = None
        self._confidence    = 0.0
        self._last_seen_time = None
        self._last_crop      = None
        self._ref_box        = None  # (center_x, center_y, size) of the last submitted face, for identity-jump detection

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def restart(self):
        """Clears the lock and starts collecting a fresh batch of predictions."""
        with self._lock:
            self._prob_history = []
            self._locked       = False
            self._label        = None
            self._confidence   = 0.0
            self._ref_box      = None

    def note_face_seen(self, seen):
        """
        Call every frame with whether a face is currently detected.
        Auto-restarts analysis if a locked result's face has been
        missing for more than FACE_LOST_RESET_SECONDS.
        """
        now = time.time()
        with self._lock:
            if seen:
                self._last_seen_time = now
            elif (self._locked and self._last_seen_time is not None and
                    now - self._last_seen_time > FACE_LOST_RESET_SECONDS):
                self._prob_history = []
                self._locked       = False
                self._label        = None
                self._confidence   = 0.0

    def submit(self, frame_bgr, landmarks, w, h):
        """
        Non-blocking. Ignored once locked, or if the worker is still
        busy. Also checks whether the face jumped position/size a lot
        since the last submission — if so, assumes a different person
        and restarts analysis rather than blending them together.
        """
        cx, cy, size = _box_center_size(landmarks, w, h)
        with self._lock:
            if self._ref_box is not None and _identity_jumped(cx, cy, size, *self._ref_box, w, h):
                if self._locked or self._prob_history:
                    print("[face_shape_model] Face position/size jumped a lot "
                          "-- assuming a new person, restarting analysis")
                self._prob_history = []
                self._locked       = False
                self._label        = None
                self._confidence   = 0.0
            self._ref_box = (cx, cy, size)

            if self._locked or self._busy:
                return
            self._pending = (frame_bgr.copy(), landmarks, w, h)
            self._busy = True

    def get_result(self):
        """Returns (label, confidence_percent, locked, collected, total)."""
        with self._lock:
            return (self._label, self._confidence, self._locked,
                    len(self._prob_history), ANALYSIS_FRAMES)

    def get_last_crop(self):
        with self._lock:
            return self._last_crop

    def is_busy(self):
        """True while the background thread is actively running inference — for a UI busy indicator."""
        with self._lock:
            return self._busy

    def stop(self):
        self._stop = True

    def _worker(self):
        while not self._stop:
            with self._lock:
                job = self._pending
                self._pending = None
            if job is None:
                time.sleep(0.01)
                continue

            frame_bgr, landmarks, w, h = job
            try:
                probs, crop = predict_shape_probs(frame_bgr, w, h)
                if probs is not None:
                    with self._lock:
                        self._last_crop = crop
                        if not self._locked:
                            self._prob_history.append(probs)
                            if len(self._prob_history) >= ANALYSIS_FRAMES:
                                self._freeze(landmarks, w, h)
            except Exception as e:
                print(f"[face_shape_model] CNN inference failed: {e}")
            finally:
                with self._lock:
                    self._busy = False

    def _freeze(self, landmarks, w, h):
        """Averages the collected predictions and locks the result. Caller must hold self._lock."""
        avg   = np.mean(self._prob_history, axis=0)
        order = np.argsort(avg)[::-1]
        top_label,    top_conf    = CLASS_NAMES[order[0]], float(avg[order[0]])
        second_label, second_conf = CLASS_NAMES[order[1]], float(avg[order[1]])

        final_label = top_label
        if (top_conf - second_conf) < TIE_BREAK_MARGIN:
            rule_based = classify_face_shape(landmarks, w, h)
            if rule_based in (top_label, second_label):
                final_label = rule_based
                print(f"[face_shape_model] Tie-break: CNN split between "
                      f"{top_label} ({top_conf:.0%}) and {second_label} ({second_conf:.0%}); "
                      f"rule-based classifier says {rule_based} -> using {rule_based}")

        self._label      = final_label
        self._confidence = float(avg[CLASS_NAMES.index(final_label)]) * 100
        self._locked     = True

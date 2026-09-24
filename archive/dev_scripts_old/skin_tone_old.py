# Old test version, superseded by main.py.
import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
from collections import Counter

# ──────────────────────────────────────────
# Download Model
# ──────────────────────────────────────────
MODEL_PATH = 'face_landmarker.task'
MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'face_landmarker/face_landmarker/float16/1/face_landmarker.task'
)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model... (first time only)")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded!")

# ──────────────────────────────────────────
# Lighting Compensation
# ──────────────────────────────────────────
def correct_lighting(frame):
    """
    Normalize lighting using LAB color space.
    Handles bright light, low light, warm/cool lighting.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_corrected = clahe.apply(l)
    corrected_lab = cv2.merge([l_corrected, a, b])
    return cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)


# ──────────────────────────────────────────
# Sample Skin Color
# ──────────────────────────────────────────
def sample_skin_color(frame, landmarks, w, h):
    """
    Sample from 5 stable face regions.
    Returns average BGR color.
    """
    sample_points = {
        "left_cheek"  : 234,
        "right_cheek" : 454,
        "forehead"    : 10,
        "nose"        : 4,
        "chin"        : 152,
    }

    colors = []
    sample_size = 10

    for name, idx in sample_points.items():
        lm = landmarks[idx]
        cx = int(lm.x * w)
        cy = int(lm.y * h)

        x1 = max(0, cx - sample_size)
        y1 = max(0, cy - sample_size)
        x2 = min(w, cx + sample_size)
        y2 = min(h, cy + sample_size)

        region = frame[y1:y2, x1:x2]
        if region.size > 0:
            median_color = np.median(
                region.reshape(-1, 3), axis=0
            )
            colors.append(median_color)

    if not colors:
        return None

    return np.mean(colors, axis=0)


# ──────────────────────────────────────────
# ITA Based Skin Tone Classification
# ──────────────────────────────────────────
def classify_skin_tone(bgr_color):
    """
    Classify using ITA angle from LAB color space.
    More accurate than RGB thresholds.
    Returns tone name, display color, ita value.
    """
    bgr   = np.uint8([[bgr_color]])
    lab   = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0][0]
    L     = float(lab[0])
    b_val = float(lab[2]) - 128

    if b_val == 0:
        ita = 90.0
    else:
        ita = np.degrees(np.arctan((L - 50) / b_val))

    if ita > 55:
        return "Very Light", (220, 210, 200), ita
    elif ita > 41:
        return "Light",      (195, 170, 150), ita
    elif ita > 28:
        return "Medium",     (160, 120, 90),  ita
    elif ita > 10:
        return "Tan",        (130, 90,  65),  ita
    elif ita > -30:
        return "Brown",      (90,  60,  40),  ita
    else:
        return "Deep",       (50,  30,  20),  ita


# ──────────────────────────────────────────
# Smart Color Recommender
# ──────────────────────────────────────────

# Full color library with BGR values
COLOR_LIBRARY = {
    "Burgundy"      : (60,   0,   128),
    "Forest Green"  : (0,    100, 0  ),
    "Navy Blue"     : (128,  0,   0  ),
    "Dusty Rose"    : (180,  130, 200),
    "Coral"         : (80,   130, 255),
    "Teal"          : (128,  128, 0  ),
    "Warm Brown"    : (43,   90,  139),
    "Olive Green"   : (0,    128, 128),
    "Mustard"       : (0,    200, 220),
    "Terracotta"    : (30,   80,  200),
    "Jewel Blue"    : (180,  80,  50 ),
    "Bright White"  : (255,  255, 255),
    "Royal Blue"    : (200,  0,   65 ),
    "Hot Pink"      : (180,  0,   255),
    "Fuchsia"       : (255,  0,   180),
    "Sky Blue"      : (235,  180, 80 ),
    "Electric Blue" : (220,  50,  50 ),
    "Gold"          : (0,    200, 255),
    "Orange"        : (0,    100, 255),
    "Yellow"        : (0,    220, 255),
    "Lavender"      : (230,  180, 200),
    "Mint Green"    : (180,  230, 180),
    "Peach"         : (150,  180, 255),
    "Cream"         : (220,  240, 255),
    "Charcoal"      : (60,   60,  60 ),
    "Emerald"       : (0,    160, 50 ),
    "Plum"          : (100,  0,   150),
    "Camel"         : (80,   140, 190),
    "Rust"          : (20,   60,  180),
    "Sage Green"    : (120,  160, 120),
}

# Colors to always avoid per tone
AVOID_COLORS = {
    "Very Light" : ["Cream", "Peach", "Bright White"],
    "Light"      : ["Peach", "Camel", "Cream"       ],
    "Medium"     : ["Camel", "Warm Brown", "Terracotta"],
    "Tan"        : ["Warm Brown", "Rust", "Camel"    ],
    "Brown"      : ["Charcoal", "Warm Brown", "Rust" ],
    "Deep"       : ["Charcoal", "Navy Blue", "Plum"  ],
}


def get_color_contrast(skin_bgr, color_bgr):
    """
    Calculate perceptual contrast between skin and outfit color.
    Uses LAB color space for human-accurate color difference (Delta E).
    Higher value = more contrast = better visibility.
    """
    skin_lab  = cv2.cvtColor(
        np.uint8([[skin_bgr]]),
        cv2.COLOR_BGR2LAB
    )[0][0].astype(float)

    color_lab = cv2.cvtColor(
        np.uint8([[color_bgr]]),
        cv2.COLOR_BGR2LAB
    )[0][0].astype(float)

    # Delta E (color difference)
    delta_e = np.sqrt(
        (skin_lab[0] - color_lab[0]) ** 2 +
        (skin_lab[1] - color_lab[1]) ** 2 +
        (skin_lab[2] - color_lab[2]) ** 2
    )

    return delta_e


def get_smart_recommendations(skin_bgr, tone):
    """
    Score every color in library based on:
    1. Perceptual contrast with actual skin color (Delta E)
    2. Not being too similar or too different
    3. Avoiding colors that clash with the tone

    This means recommendations are based on YOUR actual
    detected skin color, not just the category label.
    So even if lighting makes you appear lighter or darker,
    the suggestions still match your real skin.
    """
    avoid = AVOID_COLORS.get(tone, [])
    scored = []

    for color_name, color_bgr in COLOR_LIBRARY.items():
        # Skip avoid list
        if color_name in avoid:
            continue

        contrast = get_color_contrast(skin_bgr, color_bgr)

        # Best contrast range: 40-120 Delta E
        # Too low = too similar to skin (blends in)
        # Too high = too harsh/clashing
        if 35 <= contrast <= 130:
            scored.append((color_name, color_bgr, contrast))

    # Sort by how close to ideal contrast (80 is ideal)
    scored.sort(key=lambda x: abs(x[2] - 80))

    # Return top 6 best matches
    return scored[:6]


# ──────────────────────────────────────────
# Stable Tone Detector
# ──────────────────────────────────────────
class StableToneDetector:
    def __init__(self):
        self.history        = []
        self.bgr_history    = []
        self.HISTORY_SIZE   = 40
        self.locked_tone    = None
        self.locked_color   = None
        self.locked_bgr     = None
        self.LOCK_THRESHOLD = 30

    def update(self, tone, color, bgr):
        self.history.append(tone)
        self.bgr_history.append(bgr)

        if len(self.history) > self.HISTORY_SIZE:
            self.history.pop(0)
            self.bgr_history.pop(0)

        counts = Counter(self.history)
        most_common, count = counts.most_common(1)[0]

        if count >= self.LOCK_THRESHOLD:
            self.locked_tone  = most_common
            self.locked_color = color
            # Average BGr over history for stability
            self.locked_bgr   = np.mean(
                self.bgr_history, axis=0
            )

        return (
            self.locked_tone,
            self.locked_color,
            self.locked_bgr
        )


# ──────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────
def draw_ui(frame, tone, tone_color,
            skin_bgr, recommendations, h, w):
    """
    Draw complete UI:
    - Skin tone name at top
    - Detected skin color swatch
    - Smart recommended colors with names
    - Colors to avoid
    """

    # ── Top bar ──
    cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)

    cv2.putText(frame, f"Skin Tone: {tone}",
                (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, tone_color, 2)

    # ── Detected skin swatch ──
    sx, sy = w - 90, 8
    cv2.rectangle(frame,
                  (sx, sy), (sx + 70, sy + 40),
                  (int(skin_bgr[0]),
                   int(skin_bgr[1]),
                   int(skin_bgr[2])), -1)
    cv2.rectangle(frame,
                  (sx, sy), (sx + 70, sy + 40),
                  (255, 255, 255), 1)
    cv2.putText(frame, "Detected",
                (sx, sy - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (200, 200, 200), 1)

    # ── Bottom panel ──
    panel_h = 110
    cv2.rectangle(frame,
                  (0, h - panel_h), (w, h),
                  (20, 20, 20), -1)

    cv2.putText(frame,
                "Best Matching Colors for Your Skin:",
                (10, h - panel_h + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1)

    # ── Color swatches with names ──
    swatch_w  = 60
    swatch_h  = 35
    start_x   = 10
    swatch_y  = h - panel_h + 35

    for i, (name, bgr, contrast) in enumerate(recommendations):
        sx = start_x + i * (swatch_w + 8)

        # Color box
        cv2.rectangle(frame,
                      (sx, swatch_y),
                      (sx + swatch_w, swatch_y + swatch_h),
                      bgr, -1)
        cv2.rectangle(frame,
                      (sx, swatch_y),
                      (sx + swatch_w, swatch_y + swatch_h),
                      (255, 255, 255), 1)

        # Color name below swatch
        # Shorten long names
        short_name = name.split()[0] if len(name) > 8 else name
        cv2.putText(frame, short_name,
                    (sx, swatch_y + swatch_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (200, 200, 200), 1)

    # ── Avoid colors note ──
    avoid = AVOID_COLORS.get(tone, [])
    if avoid:
        avoid_text = "Avoid: " + ", ".join(avoid)
        cv2.putText(frame, avoid_text,
                    (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (80, 80, 255), 1)

    return frame


def draw_sample_points(frame, landmarks, w, h):
    for idx in [234, 454, 10, 4, 152]:
        lm = landmarks[idx]
        cx = int(lm.x * w)
        cy = int(lm.y * h)
        cv2.circle(frame, (cx, cy), 8, (0, 255, 255), 2)
    return frame


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
def main():
    download_model()

    BaseOptions           = mp.tasks.BaseOptions
    FaceLandmarker        = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Trying camera 1...")
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: No camera found!")
        return

    print("Camera started! Press Q to quit.")

    detector        = StableToneDetector()
    cached_recs     = []
    last_tone       = None
    avg_bgr         = np.array([180.0, 150.0, 130.0])

    with FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame     = cv2.flip(frame, 1)
            h, w, _   = frame.shape
            corrected = correct_lighting(frame)
            rgb_frame = cv2.cvtColor(
                corrected, cv2.COLOR_BGR2RGB
            )
            mp_image  = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = landmarker.detect(mp_image)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                sampled   = sample_skin_color(
                    corrected, landmarks, w, h
                )

                if sampled is not None:
                    avg_bgr = sampled
                    tone, tone_color, ita = classify_skin_tone(
                        avg_bgr
                    )
                    stable_tone, stable_color, stable_bgr = \
                        detector.update(tone, tone_color, avg_bgr)

                    # Only recalculate recommendations
                    # when tone changes — saves processing
                    if stable_tone and stable_tone != last_tone:
                        cached_recs = get_smart_recommendations(
                            stable_bgr, stable_tone
                        )
                        last_tone = stable_tone

                    frame = draw_sample_points(
                        frame, landmarks, w, h
                    )

                    if stable_tone and cached_recs:
                        frame = draw_ui(
                            frame,
                            stable_tone,
                            stable_color,
                            stable_bgr,
                            cached_recs,
                            h, w
                        )
                    else:
                        cv2.putText(
                            frame,
                            "Analyzing skin tone...",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (200, 200, 200), 2
                        )
            else:
                cv2.putText(frame,
                            "No Face - Move Closer",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

            cv2.imshow(
                "AI Grooming Assistant - Skin Tone",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
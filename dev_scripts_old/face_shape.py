# Old test version, superseded by main.py.
import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
from collections import Counter

# ──────────────────────────────────────────
# Download model if not present
# ──────────────────────────────────────────
MODEL_PATH = 'face_landmarker.task'
MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'face_landmarker/face_landmarker/float16/1/face_landmarker.task'
)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face landmark model... (first time only)")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded!")

# ──────────────────────────────────────────
# Head Pose Estimation
# ──────────────────────────────────────────
def get_head_pose(landmarks, w, h):
    """
    Estimate how much the head is turned left/right and up/down.
    Returns yaw (left/right) and pitch (up/down) in normalized units.
    """
    nose_tip    = landmarks[4]
    left_ear    = landmarks[234]
    right_ear   = landmarks[454]
    forehead    = landmarks[10]
    chin        = landmarks[152]

    # Yaw = how much head is turned left or right
    ear_center_x = (left_ear.x + right_ear.x) / 2
    yaw = abs(nose_tip.x - ear_center_x)

    # Pitch = how much head is tilted up or down
    face_center_y = (forehead.y + chin.y) / 2
    pitch = abs(nose_tip.y - face_center_y)

    return yaw, pitch

# ──────────────────────────────────────────
# Face Shape Classification
# ──────────────────────────────────────────
def classify_face_shape(landmarks, w, h):
    """
    Classify face shape using normalized coordinates so
    distance from camera does not affect the result.
    Uses head pose to only classify when face is straight.
    """

    # Check head pose first
    yaw, pitch = get_head_pose(landmarks, w, h)

    # If head is turned too much, skip classification
    if yaw > 0.08:
        return None, (200, 200, 200), "turned"

    # Use normalized coordinates (not pixel) for distance independence
    def dist(a, b):
        return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

    # Key landmarks
    forehead_top   = landmarks[10]
    chin_bottom    = landmarks[152]
    left_cheek     = landmarks[234]
    right_cheek    = landmarks[454]
    left_forehead  = landmarks[70]
    right_forehead = landmarks[300]
    left_jaw       = landmarks[58]
    right_jaw      = landmarks[288]
    left_temple    = landmarks[162]
    right_temple   = landmarks[389]

    # Measurements using normalized coordinates
    face_height     = dist(forehead_top, chin_bottom)
    cheek_width     = dist(left_cheek, right_cheek)
    forehead_width  = dist(left_forehead, right_forehead)
    jaw_width       = dist(left_jaw, right_jaw)
    temple_width    = dist(left_temple, right_temple)

    if cheek_width == 0:
        return None, (200, 200, 200), "detecting"

    # Ratios
    height_ratio   = face_height / cheek_width
    forehead_ratio = forehead_width / cheek_width
    jaw_ratio      = jaw_width / cheek_width
    temple_ratio   = temple_width / cheek_width

    # ── Classification ──
    if height_ratio > 1.55:
        shape = "Oblong"
        color = (255, 100, 0)

    elif height_ratio > 1.30:
        if forehead_ratio > 0.88 and jaw_ratio > 0.82:
            shape = "Square"
            color = (0, 100, 255)
        elif forehead_ratio > jaw_ratio + 0.12:
            shape = "Heart"
            color = (180, 0, 255)
        else:
            shape = "Oval"
            color = (0, 255, 100)

    elif height_ratio > 1.10:
        if jaw_ratio > 0.88 and forehead_ratio > 0.88:
            shape = "Square"
            color = (0, 100, 255)
        elif forehead_ratio > jaw_ratio + 0.12:
            shape = "Heart"
            color = (180, 0, 255)
        else:
            shape = "Oval"
            color = (0, 255, 100)
    else:
        shape = "Round"
        color = (0, 200, 255)

    return shape, color, "ok"

# ──────────────────────────────────────────
# Stable Shape Detector Class
# ──────────────────────────────────────────
class StableShapeDetector:
    """
    Locks onto a face shape once it is detected consistently.
    Only updates if a new shape appears very consistently.
    This prevents flickering completely.
    """
    def __init__(self):
        self.locked_shape  = None
        self.locked_color  = None
        self.history       = []
        self.HISTORY_SIZE  = 30   # frames to consider
        self.LOCK_THRESHOLD = 22  # needs 22/30 same shape to lock

    def update(self, shape, color, status):
        if status != "ok" or shape is None:
            return self.locked_shape, self.locked_color

        self.history.append(shape)
        if len(self.history) > self.HISTORY_SIZE:
            self.history.pop(0)

        # Count most common shape
        counts = Counter(self.history)
        most_common_shape, count = counts.most_common(1)[0]

        # Only lock if very consistent
        if count >= self.LOCK_THRESHOLD:
            if self.locked_shape != most_common_shape:
                self.locked_shape = most_common_shape
                self.locked_color = color

        return self.locked_shape, self.locked_color

# ──────────────────────────────────────────
# Recommendations
# ──────────────────────────────────────────
RECOMMENDATIONS = {
    "Oval": {
        "hairstyle": "Most styles suit you! Try layers or curtain bangs.",
        "glasses"  : "Most frames work! Aviators or wayfarers look great.",
    },
    "Round": {
        "hairstyle": "Try long layers or side-swept styles to add length.",
        "glasses"  : "Rectangle or square frames balance your face well.",
    },
    "Square": {
        "hairstyle": "Soft layers or waves soften your strong jawline.",
        "glasses"  : "Round or oval frames complement square faces.",
    },
    "Heart": {
        "hairstyle": "Try chin-length bobs or side parts.",
        "glasses"  : "Bottom-heavy frames like aviators balance well.",
    },
    "Oblong": {
        "hairstyle": "Try voluminous styles or curtain bangs for width.",
        "glasses"  : "Wide frames or round glasses add width to face.",
    },
}

# ──────────────────────────────────────────
# Drawing Functions
# ──────────────────────────────────────────
def draw_info(frame, shape, color, status, h, w):
    """Draw shape name, status and recommendations on frame."""

    # Status messages
    if status == "turned":
        cv2.putText(frame,
                    "Please face the camera straight",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 165, 255), 2)
    elif status == "detecting":
        cv2.putText(frame,
                    "Move closer to camera...",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
    elif shape:
        # Shape label
        cv2.putText(frame, f"Face Shape: {shape}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2)

        # Recommendations box
        rec = RECOMMENDATIONS.get(shape, {})
        if rec:
            cv2.rectangle(frame,
                          (0, h - 85),
                          (w, h),
                          (0, 0, 0), -1)
            cv2.putText(frame,
                        f"Hair: {rec['hairstyle']}",
                        (10, h - 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
            cv2.putText(frame,
                        f"Glasses: {rec['glasses']}",
                        (10, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1)
    else:
        cv2.putText(frame,
                    "Analyzing face shape...",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (200, 200, 200), 2)

    return frame


def draw_landmarks(frame, landmarks, w, h, color):
    """Draw face dots and key measurement points."""
    # All landmarks
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Key measurement points
    for idx in [10, 152, 234, 454, 70, 300, 58, 288]:
        lm = landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 5, color or (0, 255, 0), -1)

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

    # Try camera 0 then 1
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Trying camera 1...")
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: No camera found!")
        return

    print("Camera started! Press Q to quit.")
    print("Please face the camera straight for best results.")

    detector = StableShapeDetector()

    with FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame     = cv2.flip(frame, 1)
            h, w, _   = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = landmarker.detect(mp_image)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                # Classify
                shape, color, status = classify_face_shape(
                    landmarks, w, h
                )

                # Stabilize
                stable_shape, stable_color = detector.update(
                    shape, color, status
                )

                # Draw
                frame = draw_landmarks(
                    frame, landmarks, w, h, stable_color
                )
                frame = draw_info(
                    frame, stable_shape,
                    stable_color, status, h, w
                )

            else:
                cv2.putText(frame,
                            "No Face Detected - Move Closer",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

            cv2.imshow(
                "AI Grooming Assistant - Face Shape Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main()
# Old test version, superseded by main.py.
import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
from collections import Counter
import time

# ──────────────────────────────────────────
# Download Model
# ──────────────────────────────────────────
MODEL_PATH = 'face_landmarker.task'
MODEL_URL  = (
    'https://storage.googleapis.com/mediapipe-models/'
    'face_landmarker/face_landmarker/float16/1/face_landmarker.task'
)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model... (first time only)")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded!")

# ──────────────────────────────────────────
# Face Shape → Glasses Recommendation
# ──────────────────────────────────────────
FACE_SHAPE_GLASSES = {
    "Oval"    : [0, 1, 2, 3],
    "Round"   : [0, 2],
    "Square"  : [1, 3],
    "Heart"   : [0, 3],
    "Oblong"  : [1, 2],
}

FACE_SHAPE_REASON = {
    "Oval"    : "Lucky! Most frames suit oval faces.",
    "Round"   : "Rectangle frames add length to round faces.",
    "Square"  : "Round frames soften square jawlines.",
    "Heart"   : "Wider bottom frames balance heart faces.",
    "Oblong"  : "Wide frames add width to long faces.",
}

# ──────────────────────────────────────────
# Load Glasses
# ──────────────────────────────────────────
def remove_white_background(img, threshold=230):
    """Remove white/near-white background from image."""
    if img is None:
        return None
    if img.shape[2] == 4:
        return img

    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    white_mask = (
        (rgba[:, :, 0] > threshold) &
        (rgba[:, :, 1] > threshold) &
        (rgba[:, :, 2] > threshold)
    )
    rgba[white_mask, 3] = 0

    alpha = rgba[:, :, 3].astype(float)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    rgba[:, :, 3] = alpha.astype(np.uint8)
    return rgba


def load_glasses(folder='assets/glasses'):
    """Load all PNG glasses from folder."""
    glasses_list = []
    names = []

    if not os.path.exists(folder):
        print(f"ERROR: Folder '{folder}' not found!")
        return [], []

    files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])

    for filename in files:
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        img = remove_white_background(img)
        if img is not None:
            glasses_list.append(img)
            names.append(filename.replace('.png', ''))
            print(f"Loaded: {filename}")

    print(f"Total: {len(glasses_list)} glasses loaded")
    return glasses_list, names

# ──────────────────────────────────────────
# Face Shape Detection
# ──────────────────────────────────────────
def get_head_pose(landmarks, w, h):
    nose_tip = landmarks[4]
    left_ear = landmarks[234]
    right_ear = landmarks[454]
    ear_cx = (left_ear.x + right_ear.x) / 2
    yaw = abs(nose_tip.x - ear_cx)
    return yaw


def classify_face_shape(landmarks):
    """Classify face shape using normalized coordinates."""
    def dist(a, b):
        return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    face_height = dist(landmarks[10], landmarks[152])
    cheek_width = dist(landmarks[234], landmarks[454])
    forehead_width = dist(landmarks[70], landmarks[300])
    jaw_width = dist(landmarks[58], landmarks[288])

    if cheek_width == 0:
        return None

    hr = face_height / cheek_width
    fr = forehead_width / cheek_width
    jr = jaw_width / cheek_width

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
# Stable Shape Detector
# ──────────────────────────────────────────
class StableShapeDetector:
    def __init__(self):
        self.history = []
        self.locked_shape = None
        self.HISTORY_SIZE = 30
        self.LOCK_THRESHOLD = 22

    def update(self, shape):
        if shape is None:
            return self.locked_shape

        self.history.append(shape)
        if len(self.history) > self.HISTORY_SIZE:
            self.history.pop(0)

        counts = Counter(self.history)
        most_common, count = counts.most_common(1)[0]

        if count >= self.LOCK_THRESHOLD:
            self.locked_shape = most_common

        return self.locked_shape

# ──────────────────────────────────────────
# Eye Point Extraction
# ──────────────────────────────────────────
def get_eye_points(landmarks, w, h):
    # Use reliable eye landmarks
    left_eye_x = (landmarks[33].x + landmarks[133].x) / 2
    left_eye_y = (landmarks[33].y + landmarks[133].y) / 2
    right_eye_x = (landmarks[362].x + landmarks[263].x) / 2
    right_eye_y = (landmarks[362].y + landmarks[263].y) / 2
    
    lc = (int(left_eye_x * w), int(left_eye_y * h))
    rc = (int(right_eye_x * w), int(right_eye_y * h))
    
    lo = (int(landmarks[33].x * w), int(landmarks[33].y * h))
    ro = (int(landmarks[263].x * w), int(landmarks[263].y * h))
    
    return lo, ro, lc, rc

# ──────────────────────────────────────────
# LOW-LATENCY PREDICTIVE SMOOTHER
# ──────────────────────────────────────────
class PredictiveSmoother:
    """
    Combines EMA with velocity prediction for low-latency smoothing.
    Predicts where the face will be next frame based on velocity.
    """
    def __init__(self, alpha=0.4, velocity_alpha=0.3):
        self.alpha = alpha  # Smoothing factor
        self.velocity_alpha = velocity_alpha  # Velocity smoothing
        self.prev = None
        self.velocity = None
        self.last_time = time.time()
        
    def smooth(self, values):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        values = np.array(values, dtype=np.float32)
        
        if self.prev is None:
            self.prev = values.copy()
            self.velocity = np.zeros_like(values)
            return values.copy()
        
        # Calculate current velocity
        if dt > 0:
            new_velocity = (values - self.prev) / dt
            # Smooth velocity
            if self.velocity is None:
                self.velocity = new_velocity
            else:
                self.velocity = self.velocity_alpha * new_velocity + (1 - self.velocity_alpha) * self.velocity
        
        # Predict next position using velocity
        prediction = values + self.velocity * dt * 0.5  # Predict half frame ahead
        
        # EMA smooth with prediction
        smoothed = self.alpha * prediction + (1 - self.alpha) * self.prev
        
        self.prev = smoothed.copy()
        return smoothed

    def reset(self):
        self.prev = None
        self.velocity = None
        self.last_time = time.time()

# ──────────────────────────────────────────
# FAST AR Overlay - Optimized
# ──────────────────────────────────────────
def overlay_glasses(frame, glasses_img, x1, y1, gw, gh, angle):
    """
    Optimized overlay with minimal processing
    """
    fh, fw = frame.shape[:2]

    try:
        # Fast resize using INTER_LINEAR
        resized = cv2.resize(
            glasses_img, (gw, gh),
            interpolation=cv2.INTER_LINEAR
        )

        # Only rotate if angle is significant
        if abs(angle) > 1.0:
            dampened_angle = angle / 1.5
            rot_m = cv2.getRotationMatrix2D(
                (gw // 2, gh // 2),
                dampened_angle, 1.0
            )
            rotated = cv2.warpAffine(
                resized, rot_m, (gw, gh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
        else:
            rotated = resized

        # Clamp bounds
        x1_c = max(0, x1)
        y1_c = max(0, y1)
        x2_c = min(fw, x1 + gw)
        y2_c = min(fh, y1 + gh)

        if x2_c <= x1_c or y2_c <= y1_c:
            return frame

        gx1 = x1_c - x1
        gy1 = y1_c - y1
        gx2 = gx1 + (x2_c - x1_c)
        gy2 = gy1 + (y2_c - y1_c)

        if gx2 <= gx1 or gy2 <= gy1:
            return frame

        g_region = rotated[gy1:gy2, gx1:gx2]
        f_region = frame[y1_c:y2_c, x1_c:x2_c]

        if g_region.shape[:2] != f_region.shape[:2]:
            return frame

        # Fast alpha blend
        if g_region.shape[2] == 4:
            alpha = g_region[:, :, 3:4].astype(float) / 255.0
            g_bgr = g_region[:, :, :3].astype(float)
            f_bgr = f_region.astype(float)
            blended = (alpha * g_bgr + (1 - alpha) * f_bgr).astype(np.uint8)
            frame[y1_c:y2_c, x1_c:x2_c] = blended

    except Exception:
        pass

    return frame

# ──────────────────────────────────────────
# Draw UI
# ──────────────────────────────────────────
def draw_ui(frame, face_shape, current_idx,
            suggested_indices, glasses_names,
            total, h, w, fps):
    """Draw face shape, suggestion reason and controls."""

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 40), (20, 20, 20), -1)

    if face_shape:
        reason = FACE_SHAPE_REASON.get(face_shape, "Analyzing...")
        cv2.putText(
            frame,
            f"Face: {face_shape}  |  {reason}  |  FPS: {fps:.1f}",
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 150), 1
        )
    else:
        cv2.putText(
            frame, "Detecting face shape...",
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (200, 200, 200), 1
        )

    # Bottom bar
    cv2.rectangle(frame, (0, h - 50), (w, h), (20, 20, 20), -1)

    style_name = glasses_names[current_idx] if current_idx < len(glasses_names) else ""
    is_suggested = current_idx in suggested_indices
    tag = " ★ RECOMMENDED" if is_suggested else ""
    tag_color = (0, 255, 150) if is_suggested else (150, 150, 150)

    cv2.putText(
        frame,
        f"Style {current_idx + 1}/{total}: {style_name}{tag}",
        (10, h - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, tag_color, 1
    )

    cv2.putText(
        frame,
        "N = Next  |  P = Prev  |  Q = Quit",
        (10, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4, (120, 120, 120), 1
    )

    return frame

# ──────────────────────────────────────────
# Main - OPTIMIZED
# ──────────────────────────────────────────
def main():
    download_model()

    glasses_list, glasses_names = load_glasses('assets/glasses')
    if not glasses_list:
        print("No glasses found!")
        return

    current_idx = 0
    total = len(glasses_list)

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: No camera found!")
        return

    # Optimize camera for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency

    print("Camera started!")
    print("N = Next | P = Previous | Q = Quit")

    # Use predictive smoother for lower latency
    smoother = PredictiveSmoother(alpha=0.35, velocity_alpha=0.3)
    shape_detector = StableShapeDetector()
    face_shape = None
    suggested = list(range(total))
    frame_count = 0
    
    # FPS tracking
    fps = 0
    fps_counter = 0
    fps_timer = time.time()

    with FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Process every frame but with optimizations
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = landmarker.detect(mp_image)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                # Face shape detection (less frequent)
                frame_count += 1
                if frame_count % 15 == 0:
                    yaw = get_head_pose(landmarks, w, h)
                    if yaw < 0.08:
                        raw_shape = classify_face_shape(landmarks)
                        face_shape = shape_detector.update(raw_shape)
                        if face_shape:
                            suggested = FACE_SHAPE_GLASSES.get(face_shape, list(range(total)))

                # Eye positions
                lo, ro, lc, rc = get_eye_points(landmarks, w, h)

                # Calculate eye distance
                eye_dist = abs(ro[0] - lo[0])
                
                # Glasses width
                gw = int(eye_dist * 1.6)
                if gw < 10:
                    continue

                # Maintain aspect ratio
                orig_h, orig_w = glasses_list[current_idx].shape[:2]
                aspect = orig_h / orig_w
                gh = int(gw * aspect)

                # Center between eyes
                cx = (lc[0] + rc[0]) // 2
                cy = (lc[1] + rc[1]) // 2

                # Angle (head tilt)
                angle = np.degrees(np.arctan2(rc[1] - lc[1], rc[0] - lc[0]))

                # Position
                x1 = cx - gw // 2
                y1 = cy - int(gh * 0.5)

                # Smooth with prediction
                values = np.array([x1, y1, gw, gh, angle], dtype=float)
                smoothed = smoother.smooth(values)

                sx1 = int(smoothed[0])
                sy1 = int(smoothed[1])
                sgw = int(smoothed[2])
                sgh = int(smoothed[3])
                sang = float(smoothed[4])

                # Overlay glasses
                frame = overlay_glasses(
                    frame,
                    glasses_list[current_idx],
                    sx1, sy1, sgw, sgh,
                    sang
                )

            else:
                smoother.reset()
                cv2.putText(
                    frame,
                    "No Face - Move Closer",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2
                )

            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_timer = time.time()

            # Draw UI
            frame = draw_ui(
                frame, face_shape,
                current_idx, suggested,
                glasses_names, total, h, w, fps
            )

            cv2.imshow("AI Grooming Assistant - AR Glasses", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                current_idx = (current_idx + 1) % total
                smoother.reset()
            elif key == ord('p'):
                current_idx = (current_idx - 1) % total
                smoother.reset()

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
"""
MediaPipe face landmark setup and extraction, plus the signal-
smoothing helpers used on top of raw per-frame landmark data
(OneEuroFilter for continuous values like glasses position/angle,
StableDetector for discrete labels like face shape/skin tone), and
3D head-pose estimation (solve_head_pose) for the AR glasses.
"""
import math
import mediapipe as mp
import numpy as np
import cv2
import urllib.request
import os
import time
from collections import Counter

from app.paths import MODELS_DIR

# ──────────────────────────────────────────
# Download Model
# ──────────────────────────────────────────
MODEL_PATH = str(MODELS_DIR / 'face_landmarker.task')
MODEL_URL  = (
    'https://storage.googleapis.com/mediapipe-models/'
    'face_landmarker/face_landmarker/float16/1/'
    'face_landmarker.task'
)


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model ready!")


def create_landmarker():
    """
    Builds a ready-to-use MediaPipe FaceLandmarker in VIDEO mode.
    Use as a context manager: `with create_landmarker() as landmarker:`
    """
    download_model()

    BaseOptions           = mp.tasks.BaseOptions
    FaceLandmarker        = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        # Detect up to 3 faces (not just 1) so that when more than one
        # person is in frame we can pick the largest ourselves and
        # tell the user, instead of silently tracking whichever face
        # MediaPipe's internal detector happened to rank first.
        num_faces=3,
    )
    return FaceLandmarker.create_from_options(options)


class FrameTimestamper:
    """
    MediaPipe VIDEO mode needs a timestamp (ms) that strictly increases
    frame to frame. Uses real elapsed time rather than an assumed fixed
    fps, since actual per-frame processing time varies.
    """
    def __init__(self):
        self._start = time.perf_counter()
        self._last  = -1

    def next(self):
        ts = int((time.perf_counter() - self._start) * 1000)
        if ts <= self._last:
            # guard MediaPipe's requirement that each timestamp be
            # strictly greater than the last one
            ts = self._last + 1
        self._last = ts
        return ts


def _bbox_area(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def detect_landmarks(landmarker, timestamper, rgb_frame):
    """
    Runs MediaPipe on one RGB frame. Returns (landmarks, multiple_faces):
    landmarks is the largest detected face (by landmark bounding-box
    area), or None if no face was found; multiple_faces is True if
    more than one face was in frame, so the caller can tell the user
    which one it picked.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect_for_video(mp_image, timestamper.next())
    if not result.face_landmarks:
        return None, False
    if len(result.face_landmarks) == 1:
        return result.face_landmarks[0], False
    largest = max(result.face_landmarks, key=_bbox_area)
    return largest, True


def get_head_yaw(landmarks):
    nose      = landmarks[4]
    left_ear  = landmarks[234]
    right_ear = landmarks[454]
    ear_cx    = (left_ear.x + right_ear.x) / 2
    return abs(nose.x - ear_cx)


# ──────────────────────────────────────────
# 3D head pose (yaw/pitch/roll) via cv2.solvePnP, for the AR glasses'
# perspective warp. Uses the classic 6-point generic face model (mm,
# arbitrary but self-consistent scale) matched to these MediaPipe
# 468-landmark indices -- this exact point set/index pairing is the
# standard recipe used across most OpenCV+MediaPipe head-pose code:
#   1   = nose tip
#   152 = chin
#   33  = left eye, outer corner
#   263 = right eye, outer corner
#   61  = left mouth corner
#   291 = right mouth corner
# ──────────────────────────────────────────
POSE_LANDMARK_IDXS = (1, 152, 33, 263, 61, 291)
# Y and Z are negated relative to the "textbook" version of this model
# (which uses a Y-up, Z-toward-viewer convention) to match OpenCV's
# camera convention (Y down, Z into the scene) -- without this, a face
# looking straight at the camera solves to a ~180 degree rotation
# instead of ~0, which is harmless for the flat-rectangle warp itself
# (confirmed by rendering a real test photo) but risks the pitch value
# wrapping around near +-180 degrees, which would destabilize the One
# Euro filter. Verified empirically against a real photo run through
# MediaPipe: this flip keeps yaw/roll numerically identical and only
# re-centers pitch.
POSE_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, 330.0, 65.0),        # Chin
    (-225.0, -170.0, 135.0),   # Left eye, outer corner
    (225.0, -170.0, 135.0),    # Right eye, outer corner
    (-150.0, 150.0, 125.0),    # Left mouth corner
    (150.0, 150.0, 125.0),     # Right mouth corner
], dtype=np.float64)


def rotation_matrix_to_euler(R):
    """
    3x3 rotation matrix -> (pitch, yaw, roll) in radians, for R defined
    as Rz(roll) @ Ry(yaw) @ Rx(pitch) (the standard XYZ-intrinsic
    decomposition). Paired with euler_to_rotation_matrix below for a
    faithful round-trip -- verified in this module's self-test, since
    the OneEuroFilter smoothing path depends on decompose -> smooth ->
    reconstruct reproducing the original rotation.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy >= 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal-lock edge case (looking straight up/down)
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = 0.0
    return pitch, yaw, roll


def euler_to_rotation_matrix(pitch, yaw, roll):
    """Inverse of rotation_matrix_to_euler."""
    rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch), math.cos(pitch)]])
    ry = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                   [0, 1, 0],
                   [-math.sin(yaw), 0, math.cos(yaw)]])
    rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                   [math.sin(roll), math.cos(roll), 0],
                   [0, 0, 1]])
    return rz @ ry @ rx


def _camera_matrix(w, h):
    """Approximate camera intrinsics (no real calibration available for a webcam) -- the standard focal~=width, centered-principal-point approximation used throughout OpenCV head-pose tutorials."""
    return np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)


_DIST_COEFFS = np.zeros((4, 1))


def solve_head_pose(landmarks, w, h):
    """
    Estimates head pose from 6 stable landmarks via cv2.solvePnP.
    Returns (ok, rvec, tvec, (pitch_deg, yaw_deg, roll_deg)).
    ok is False (with the other fields None) if solvePnP fails or
    throws -- callers should treat that like "no reliable pose" (e.g.
    hide the AR glasses that frame) rather than crash.
    """
    try:
        image_points = np.array(
            [(landmarks[i].x * w, landmarks[i].y * h) for i in POSE_LANDMARK_IDXS],
            dtype=np.float64,
        )
        ok, rvec, tvec = cv2.solvePnP(
            POSE_MODEL_POINTS, image_points, _camera_matrix(w, h), _DIST_COEFFS,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return False, None, None, None
        R, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = rotation_matrix_to_euler(R)
        return True, rvec, tvec, (math.degrees(pitch), math.degrees(yaw), math.degrees(roll))
    except Exception:
        return False, None, None, None


def _selftest():
    """Synthetic, no-camera-needed correctness check for the Euler round-trip and solvePnP sign/magnitude -- run via `python -m app.face_tracking` from the project root."""
    w, h = 640, 480
    cam = _camera_matrix(w, h)
    for pitch, yaw, roll in [(0.1, 0.3, -0.2), (0.0, 0.0, 0.0), (-0.4, 0.6, 0.15)]:
        R = euler_to_rotation_matrix(pitch, yaw, roll)
        p2, y2, r2 = rotation_matrix_to_euler(R)
        R2 = euler_to_rotation_matrix(p2, y2, r2)
        assert np.max(np.abs(R - R2)) < 1e-9, "Euler round-trip mismatch"

    true_tvec = np.array([[0.0], [0.0], [600.0]])
    for known_yaw_deg in (-30, -10, 0, 25, 34):
        R_true = euler_to_rotation_matrix(0.0, math.radians(known_yaw_deg), 0.0)
        rvec_true, _ = cv2.Rodrigues(R_true)
        projected, _ = cv2.projectPoints(POSE_MODEL_POINTS, rvec_true, true_tvec, cam, _DIST_COEFFS)
        landmarks = [None] * 468
        for idx, pt in zip(POSE_LANDMARK_IDXS, projected.reshape(-1, 2)):
            landmarks[idx] = type('L', (), {'x': pt[0] / w, 'y': pt[1] / h})()
        ok, _, _, (pitch, yaw, roll) = solve_head_pose(landmarks, w, h)
        assert ok, "solve_head_pose failed on a clean synthetic pose"
        assert abs(yaw - known_yaw_deg) < 1.0, f"yaw {yaw} != expected {known_yaw_deg}"
    print("[face_tracking] selftest passed")


if __name__ == "__main__":
    _selftest()


class StableDetector:
    """
    Debounces a noisy per-frame discrete label (e.g. face shape, skin
    tone) so the UI only updates once a value has been the most common
    answer over the last `size` updates at least `threshold` times.
    """
    def __init__(self, size=30, threshold=22):
        self.history   = []
        self.locked    = None
        self.size      = size
        self.threshold = threshold

    def update(self, value):
        if value is None:
            return self.locked
        self.history.append(value)
        if len(self.history) > self.size:
            self.history.pop(0)
        counts = Counter(self.history)
        top, count = counts.most_common(1)[0]
        if count >= self.threshold:
            self.locked = top
        return self.locked


class OneEuroFilter:
    """
    Smooths noisy per-frame values (glasses x/y/size/angle) so they
    don't shake when the head is still, but still keep up when the
    head moves fast. Unlike a fixed-ratio average, the smoothing
    strength here adapts every frame based on how fast the value is
    currently changing (Casiez et al., "1€ Filter", 2012).
    """
    def __init__(self, min_cutoff=1.2, beta=3.0, d_cutoff=1.0):
        # min_cutoff: smoothing strength when the value is nearly
        # still. Lower = steadier but more lag if motion starts.
        self.min_cutoff = min_cutoff
        # beta: how much a fast-changing value is allowed to cut
        # through the smoothing. Raised from 0.7 to 3.0 to chase
        # near-zero lag while moving; only affects moving frames
        # (dx≈0 while still, so this doesn't bring back jitter).
        self.beta = beta
        # d_cutoff: smooths the speed estimate itself, so a single
        # noisy frame-to-frame jump doesn't spike the reactivity.
        self.d_cutoff = d_cutoff
        self.x_prev  = None
        self.dx_prev = None
        self.t_prev  = None

    @staticmethod
    def _alpha(t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def smooth(self, values, t=None):
        if t is None:
            t = time.time()
        if self.x_prev is None:
            self.x_prev  = values.copy()
            self.dx_prev = np.zeros_like(values)
            self.t_prev  = t
            return values.copy()

        t_e = max(t - self.t_prev, 1e-6)  # guard divide-by-zero

        a_d     = self._alpha(t_e, self.d_cutoff)
        dx      = (values - self.x_prev) / t_e
        dx_hat  = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a      = self._alpha(t_e, cutoff)
        x_hat  = a * values + (1 - a) * self.x_prev

        self.x_prev  = x_hat
        self.dx_prev = dx_hat
        self.t_prev  = t
        return x_hat.copy()

    def reset(self):
        self.x_prev  = None
        self.dx_prev = None
        self.t_prev  = None

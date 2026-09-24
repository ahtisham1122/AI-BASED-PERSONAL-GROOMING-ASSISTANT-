"""
AR glasses: asset loading (auto-crop + lens-anchor detection + edge
feathering, all done ONCE at load and cached — never per frame) and
the perspective-warp overlay that places them using real head pose.

3D placement note: cv2.solvePnP's rotation/translation are estimated
in a generic mm-scale face model (see face_tracking.POSE_MODEL_POINTS)
that has no fixed relationship to this app's actual pixel measurements
(no real camera calibration exists for an arbitrary webcam). Rather
than mixing those two unit systems, the ROTATION from solvePnP is
applied to a flat rectangle sized/positioned from this frame's own
real pixel landmarks (temple width, nose bridge, eye centers — see
compute_glasses_geometry), then projected back to 2D with a simple
distance-based scale falloff (_project_pseudo3d) tuned by FOCAL_RATIO.
This is a deliberate, documented approximation: a fully metric
perspective projection would need a calibrated camera, which a random
webcam doesn't have.
"""
import cv2
import numpy as np
import os
import json
import math
from collections import namedtuple

from app.paths import ASSETS_DIR

GlassesAsset = namedtuple('GlassesAsset', ['img', 'anchor_left', 'anchor_right', 'name'])

ANCHORS_PATH = str(ASSETS_DIR / 'glasses' / 'glasses_anchors.json')

# How pronounced the perspective foreshortening looks as the head
# turns (see module docstring — there's no real camera calibration to
# derive this from, it's a tunable "how dramatic the 3D effect looks"
# knob, similar to picking a virtual field of view). Higher = subtler.
FOCAL_RATIO = 2.4


def remove_white_bg(img, threshold=230, falloff=25, blur_ksize=7):
    """
    Makes a white/near-white background transparent. Uses a soft
    (ramped) threshold instead of a hard cutoff, then feathers only
    the boundary band with a blur, so the glasses-to-transparent
    edge doesn't leave a hard white/gray halo. PNGs that already
    ship with real transparency are left untouched.
    """
    if img is None:
        return None

    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        # "Real" alpha: some pixels are meaningfully transparent —
        # not just a fully-opaque channel the exporter tacked on.
        # 250/0.01 are loose slack for near-opaque compression noise.
        has_real_alpha = (
            alpha.min() < 250 and (alpha < 250).mean() > 0.01
        )
        if has_real_alpha:
            return img
        bgr = img[:, :, :3]
    else:
        bgr = img

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(float)

    # Soft threshold: fully opaque `falloff` levels below `threshold`,
    # fully transparent at/above it, linearly ramped in between —
    # replaces the old hard >threshold cutoff that caused the halo.
    alpha = np.clip(
        (threshold - gray) / falloff * 255, 0, 255
    ).astype(np.uint8)

    # Blur only the boundary band (where alpha is actually changing),
    # so flat opaque/transparent areas stay crisp and only the real
    # edge transition gets softened.
    edge_band = cv2.morphologyEx(
        alpha, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)
    ) > 0
    blurred = cv2.GaussianBlur(alpha, (blur_ksize, blur_ksize), 0)
    alpha = np.where(edge_band, blurred, alpha).astype(np.uint8)

    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return rgba


def _auto_crop(rgba, alpha_thresh=10, pad=2):
    """Crops to the bounding box of visible (non-transparent) content — every PNG ships with different empty margins."""
    alpha = rgba[:, :, 3]
    mask = alpha > alpha_thresh
    if not mask.any():
        return rgba
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - pad), min(rgba.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(rgba.shape[1], xs.max() + 1 + pad)
    return rgba[y0:y1, x0:x1]


def _feather_alpha(rgba, blur_ksize=5):
    """One-time soft edge on the alpha channel (same edge-band-blur trick as remove_white_bg) so the warped glasses never show a hard outline/halo, regardless of how clean the source PNG's own edge was."""
    alpha = rgba[:, :, 3]
    edge_band = cv2.morphologyEx(alpha, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    blurred = cv2.GaussianBlur(alpha, (blur_ksize, blur_ksize), 0)
    out = rgba.copy()
    out[:, :, 3] = np.where(edge_band, blurred, alpha).astype(np.uint8)
    return out


def _estimate_anchor_pair(w, h):
    """Fallback lens-center guess when a frame has no hand-tuned entry: left/right third of the width, vertical middle."""
    return (w / 3.0, h / 2.0), (w * 2.0 / 3.0, h / 2.0)


def _load_anchor_file():
    if not os.path.exists(ANCHORS_PATH):
        return {}
    try:
        with open(ANCHORS_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"[ar_overlay] Couldn't read {ANCHORS_PATH} ({e}); starting fresh.")
        return {}


def _save_anchor_file(data):
    try:
        os.makedirs(os.path.dirname(ANCHORS_PATH), exist_ok=True)
        with open(ANCHORS_PATH, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"[ar_overlay] Couldn't save {ANCHORS_PATH}: {e}")


def load_glasses(folder=str(ASSETS_DIR / 'glasses')):
    """
    Loads every PNG in `folder`, auto-crops each to its visible
    content, feathers the alpha edge, and resolves lens-anchor
    (left/right lens-center) positions from assets/glasses/
    glasses_anchors.json — estimating and WRITING BACK an entry for
    any frame that doesn't have one yet, so it can be hand-tuned
    later. All of this happens once here, not per frame (see
    GlassesAsset — the cached, ready-to-warp image and its anchors).
    """
    glasses_list, names = [], []
    if not os.path.exists(folder):
        return [], []

    anchors_data = _load_anchor_file()
    changed = False
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith('.png'))

    for f in files:
        raw = cv2.imread(os.path.join(folder, f), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        rgba = remove_white_bg(raw)
        if rgba is None or rgba.shape[2] != 4:
            continue
        cropped = _feather_alpha(_auto_crop(rgba))

        name = os.path.splitext(f)[0]
        entry = anchors_data.get(name)
        if entry and 'left' in entry and 'right' in entry:
            anchor_l = tuple(entry['left'])
            anchor_r = tuple(entry['right'])
        else:
            ch, cw = cropped.shape[:2]
            anchor_l, anchor_r = _estimate_anchor_pair(cw, ch)
            anchors_data[name] = {'left': list(anchor_l), 'right': list(anchor_r)}
            changed = True

        glasses_list.append(GlassesAsset(cropped, anchor_l, anchor_r, name))
        names.append(name)

    if changed:
        anchors_data['_note'] = (
            "left/right are lens-CENTER pixel coordinates in each frame's own "
            "auto-cropped image (origin top-left) -- edit by hand to fine-tune "
            "lens alignment, then just restart the app."
        )
        _save_anchor_file(anchors_data)

    return glasses_list, names


def _lm_px(lm, w, h):
    return (lm.x * w, lm.y * h)


def compute_eye_targets_px(landmarks, w, h):
    """(left_eye_px, right_eye_px) -- iris centers (468/473) when available, else the eye-corner midpoints."""
    if len(landmarks) > 468:
        return _lm_px(landmarks[468], w, h), _lm_px(landmarks[473], w, h)
    lx = (landmarks[33].x + landmarks[133].x) / 2 * w
    ly = (landmarks[33].y + landmarks[133].y) / 2 * h
    rx = (landmarks[362].x + landmarks[263].x) / 2 * w
    ry = (landmarks[362].y + landmarks[263].y) / 2 * h
    return (lx, ly), (rx, ry)


def compute_glasses_geometry(landmarks, w, h):
    """
    Real per-frame measurements (float pixels, no rounding): glasses
    width from the temple landmarks x1.05, horizontal center from the
    nose bridge, vertical center from the eye line. Returns
    (gw, center_x, center_y) -- exactly the 3 values that get smoothed
    (see main.py's pos_filter).
    """
    temple_l = _lm_px(landmarks[127], w, h)
    temple_r = _lm_px(landmarks[356], w, h)
    gw = math.hypot(temple_r[0] - temple_l[0], temple_r[1] - temple_l[1]) * 1.05
    nose_bridge = _lm_px(landmarks[168], w, h)
    eye_l, eye_r = compute_eye_targets_px(landmarks, w, h)
    center_x = nose_bridge[0]
    center_y = (eye_l[1] + eye_r[1]) / 2.0
    return gw, center_x, center_y


def _rect_corners_local(gw, gh):
    """The 4 corners of a flat gw x gh rectangle, centered at its own origin (z=0)."""
    hw, hh = gw / 2.0, gh / 2.0
    return np.array([
        [-hw, -hh, 0.0],
        [hw, -hh, 0.0],
        [hw, hh, 0.0],
        [-hw, hh, 0.0],
    ], dtype=np.float64)


def _project_pseudo3d(points_local, focal):
    """Rotated 3D points (still centered at local origin) -> 2D points, still centered at local origin, with a simple depth-based scale falloff (see module docstring)."""
    z = points_local[:, 2]
    scale = focal / (focal + z)
    return np.stack([points_local[:, 0] * scale, points_local[:, 1] * scale], axis=1)


def build_glasses_corners(asset, gw, center, R):
    """
    Returns (src_corners, dst_corners), float64 4x2 each, for
    cv2.getPerspectiveTransform: src is the cached asset's own 4
    corners, dst is where they land this frame.

    The rectangle is sized from the (smoothed) real per-person width
    gw and rotated by the (smoothed) head pose R same as before, but
    positioning now goes through the asset's lens ANCHORS rather than
    its raw geometric center: every glasses PNG has a different amount
    of frame/temple below the lenses, so centering the bounding box
    doesn't put the lenses over the eyes -- rotating the anchor point
    along with the corners and then shifting so the (rotated) anchor
    lands exactly on `center` does.
    """
    sh, sw = asset.img.shape[:2]
    k = gw / sw
    gh = sh * k
    focal = gw * FOCAL_RATIO

    img_cx, img_cy = sw / 2.0 * k, sh / 2.0 * k
    anchor_local = np.array([[
        (asset.anchor_left[0] + asset.anchor_right[0]) / 2.0 * k - img_cx,
        (asset.anchor_left[1] + asset.anchor_right[1]) / 2.0 * k - img_cy,
        0.0,
    ]])

    corners_local = _rect_corners_local(gw, gh)
    rotated = np.vstack([corners_local, anchor_local]) @ R.T  # (5, 3): 4 corners + anchor, rotated together
    projected = _project_pseudo3d(rotated, focal)  # (5, 2), still centered at local origin

    shift = np.array(center) - projected[4]  # move the (rotated) anchor exactly onto the target center
    dst = projected[:4] + shift

    src = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float64)
    return src, dst


def warp_and_blend_glasses(frame, asset, src_corners, dst_corners, alpha_mult):
    """
    Warps the cached asset onto `frame` with a perspective transform
    (src_corners -> dst_corners, both float/sub-pixel), INTER_LINEAR-
    interpolated, alpha-blended by alpha_mult (the fade in/out factor).
    Mutates and returns frame.

    Only warps/blends a small canvas around the glasses' own bounding
    box, not the whole frame -- warping+blending the full frame every
    frame measured ~15ms (mostly the blend), which alone would blow
    the frame budget; this cuts it to the glasses' actual footprint.
    The canvas's pixel origin (x0, y0) is floored to an int (numpy
    slicing needs that), but the sub-pixel fractional part of each
    corner is preserved exactly in the translated transform -- it's
    baked into the warp, not rounded away, so this doesn't reintroduce
    the 1px jitter that placing at rounded coordinates would.
    """
    if alpha_mult <= 0.001:
        return frame
    fh, fw = frame.shape[:2]
    try:
        margin = 4  # a few px so the feathered edge has room, avoids a hard clip seam at the canvas border
        x0 = max(0, int(np.floor(dst_corners[:, 0].min())) - margin)
        y0 = max(0, int(np.floor(dst_corners[:, 1].min())) - margin)
        x1 = min(fw, int(np.ceil(dst_corners[:, 0].max())) + margin)
        y1 = min(fh, int(np.ceil(dst_corners[:, 1].max())) + margin)
        if x1 <= x0 or y1 <= y0:
            return frame

        local_dst = dst_corners - np.array([x0, y0])
        M = cv2.getPerspectiveTransform(src_corners.astype(np.float32), local_dst.astype(np.float32))
        warped = cv2.warpPerspective(
            asset.img, M, (x1 - x0, y1 - y0),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
        )
        roi = frame[y0:y1, x0:x1]
        alpha = (warped[:, :, 3:4].astype(np.float32) / 255.0) * alpha_mult
        frame[y0:y1, x0:x1] = (alpha * warped[:, :, :3] + (1 - alpha) * roi.astype(np.float32)).astype(np.uint8)
    except Exception:
        pass
    return frame

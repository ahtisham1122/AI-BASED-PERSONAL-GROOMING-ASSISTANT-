"""AR glasses: loading assets, background removal, and overlaying onto the frame."""
import cv2
import numpy as np
import os


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


def load_glasses(folder='assets/glasses'):
    glasses, names = [], []
    if not os.path.exists(folder):
        return [], []
    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith('.png')
    ])
    for f in files:
        img = cv2.imread(
            os.path.join(folder, f),
            cv2.IMREAD_UNCHANGED
        )
        if img is not None:
            img = remove_white_bg(img)
            glasses.append(img)
            names.append(f.replace('.png', ''))
    return glasses, names


def overlay_glasses(frame, glasses_img,
                    x1, y1, gw, gh, angle):
    fh, fw = frame.shape[:2]
    try:
        resized = cv2.resize(
            glasses_img, (gw, gh),
            interpolation=cv2.INTER_AREA
        )
        # atan2 on image pixels (y grows downward) gives an angle
        # with the opposite sign of what cv2.getRotationMatrix2D
        # expects (it treats positive as counter-clockwise on
        # screen) — negate it so the glasses tilt the same way the
        # head does. Full-strength rotation, no damping.
        rot_m = cv2.getRotationMatrix2D(
            (gw // 2, gh // 2),
            -angle, 1.0
        )
        rotated = cv2.warpAffine(
            resized, rot_m, (gw, gh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(fw, x1 + gw)
        y2c = min(fh, y1 + gh)
        if x2c <= x1c or y2c <= y1c:
            return frame
        gx1 = x1c - x1
        gy1 = y1c - y1
        gx2 = gx1 + (x2c - x1c)
        gy2 = gy1 + (y2c - y1c)
        gr  = rotated[gy1:gy2, gx1:gx2]
        fr  = frame[y1c:y2c, x1c:x2c]
        if gr.shape[:2] != fr.shape[:2]:
            return frame
        if gr.shape[2] == 4:
            alpha   = gr[:, :, 3:4].astype(
                float) / 255.0
            blended = (
                alpha * gr[:, :, :3].astype(float) +
                (1 - alpha) * fr.astype(float)
            ).astype(np.uint8)
            frame[y1c:y2c, x1c:x2c] = blended
    except Exception:
        pass
    return frame

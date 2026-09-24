"""Skin tone sampling and classification."""
import cv2
import numpy as np


def correct_lighting(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    l = clahe.apply(l)
    return cv2.cvtColor(
        cv2.merge([l, a, b]),
        cv2.COLOR_LAB2BGR
    )


# If every sampled point is this dark or this bright, the frame is
# likely in near-total darkness or blown out by backlighting — the
# color read back in that regime is more a property of the exposure
# than the skin, so it's better to skip the sample than confidently
# report a tone from bad data.
MIN_RELIABLE_BRIGHTNESS = 15
MAX_RELIABLE_BRIGHTNESS = 240


# How much brighter the brightest sample point can be than the darkest
# before we treat it as a partial shadow rather than normal skin/lighting
# variation (e.g. one cheek noticeably darker than the other because a
# cap brim, hair, or angled light is covering part of the face). 1.6
# means "60% brighter" -- picked to tolerate everyday lighting unevenness
# without flagging every photo, while still catching a clearly one-sided
# shadow.
SHADOW_FLAG_RATIO = 1.6


def _sample_patch(frame, landmarks, idx, w, h):
    """Median BGR color in a 20x20 px patch around one landmark, or None if it's off-frame."""
    lm = landmarks[idx]
    cx = int(lm.x * w)
    cy = int(lm.y * h)
    x1 = max(0, cx - 10)
    y1 = max(0, cy - 10)
    x2 = min(w, cx + 10)
    y2 = min(h, cy + 10)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return None
    return np.median(region.reshape(-1, 3), axis=0)


def sample_skin_color(frame, landmarks, w, h):
    """
    Samples skin color from the cheeks and chin -- deliberately avoiding
    the forehead/hairline, since that's the first place a cap or hat
    brim shadows or covers the skin outright.

    Returns (avg_bgr, low_confidence):
      avg_bgr        the sampled color, or None if no sample point was
                      usable at all, or if the overall sample was too
                      dark/bright to trust (see MIN/MAX_RELIABLE_BRIGHTNESS).
      low_confidence  True if the left cheek and right cheek came back
                      at noticeably different brightness (see
                      SHADOW_FLAG_RATIO) -- a sign that something is
                      shadowing one side of the face (a tilted cap brim,
                      hair, a light off to one side) rather than the
                      skin genuinely being that color. Comparing left
                      vs. right specifically (rather than just the
                      darkest vs. brightest of all points) avoids false
                      alarms from ordinary top-to-bottom face shading --
                      e.g. the chin reads a bit darker than the cheeks
                      under most overhead lighting even with no cap
                      involved, which isn't a sign of anything wrong.
    """
    # 234/454: upper cheek, level with the cheekbones (same landmarks
    # face_shape_model.py uses for cheekbone width).
    # 50/280: lower cheek -- the fleshy part below the eyes, well above
    # the jaw -- clearly below where any cap brim or hairline would reach.
    # Paired up (234 with 50, 454 with 280) as this side's two samples,
    # so the shadow check below compares "this side of the face" against
    # "that side", not individual points against each other.
    left_idxs  = [234, 50]
    right_idxs = [454, 280]
    chin_idx   = 152

    left_colors  = [c for c in (_sample_patch(frame, landmarks, i, w, h) for i in left_idxs)  if c is not None]
    right_colors = [c for c in (_sample_patch(frame, landmarks, i, w, h) for i in right_idxs) if c is not None]
    chin_color   = _sample_patch(frame, landmarks, chin_idx, w, h)

    all_colors = left_colors + right_colors + ([chin_color] if chin_color is not None else [])
    if not all_colors:
        return None, False

    avg = np.mean(all_colors, axis=0)
    brightness = np.mean(avg)
    if brightness < MIN_RELIABLE_BRIGHTNESS or brightness > MAX_RELIABLE_BRIGHTNESS:
        return None, False

    low_confidence = False
    if left_colors and right_colors:
        left_brightness  = np.mean([np.mean(c) for c in left_colors])
        right_brightness = np.mean([np.mean(c) for c in right_colors])
        darker, brighter = sorted((left_brightness, right_brightness))
        # bool(...) so callers (including Flask's JSON encoder, which
        # chokes on numpy's bool_ type) get a plain Python bool.
        low_confidence = bool(darker > 0 and (brighter / darker) > SHADOW_FLAG_RATIO)

    return avg, low_confidence


def classify_skin_tone(bgr_color):
    bgr = np.uint8([[bgr_color]])
    lab = cv2.cvtColor(
        bgr, cv2.COLOR_BGR2LAB
    )[0][0]
    # OpenCV's 8-bit LAB conversion returns L in [0, 255], but the
    # standard ITA (Individual Typology Angle) formula below assumes
    # the textbook CIE L* range of [0, 100] — without this scaling,
    # (L-50) is roughly 2.5x too large and swamps b_val, collapsing
    # almost every real skin sample into "Very Light" regardless of
    # actual tone (verified: raw L of 137-181 gave ITA 77-84 for every
    # test photo; correctly scaled, the same photos spread 14-55).
    L     = float(lab[0]) * 100.0 / 255.0
    b_val = float(lab[2]) - 128
    ita   = np.degrees(
        np.arctan((L - 50) / b_val)
    ) if b_val != 0 else 90.0
    print(f"[skin_tone] sampled_bgr={tuple(round(float(c), 1) for c in bgr_color)} ITA={ita:.1f}")

    if ita > 55:   return "Very Light"
    elif ita > 41: return "Light"
    elif ita > 28: return "Medium"
    elif ita > 10: return "Tan"
    elif ita > -30:return "Brown"
    else:          return "Deep"

"""
On-screen UI: a 1280x720 layout with the camera feed on the left and a
tab-driven card panel on the right, rendered with PIL (TrueType text,
rounded cards, gradients) and composited onto the OpenCV frame.

Visual language: near-black navy background, blue accent for
face-shape/male-related elements, purple/pink accent for
skin-tone/female-related elements.

Smoothness: PIL's rounded_rectangle/ellipse/polygon primitives are NOT
anti-aliased on their own -- drawing them at 1x leaves visibly jagged
("staircase") edges. Every cached image in this file (top bar, right
panel, bottom bar, small badges, the gender screen) is therefore drawn
at SS-times its final size and downsampled once with LANCZOS
resampling before being cached -- see SS below. This only costs
anything when a cache is actually rebuilt (on a state change), never
per frame. The live camera border/overlays, which genuinely repaint
every frame, use plain cv2 primitives with cv2.LINE_AA instead (cheap,
and cv2's own anti-aliasing) rather than any PIL supersampling.

Performance: the top bar, right panel, and bottom bar are each cached
as plain BGR images and only re-rendered when the data they show
actually changes; every frame just pastes the cached images in with
numpy slicing. The video feed's rounded corners use small (radius x
radius) boolean masks applied only to the 4 corner squares -- not the
whole frame -- so they cost microseconds even though they run every
frame; the border itself is drawn every frame with cv2.ellipse/line
(anti-aliased, cheap) rather than any per-frame blur. Only the video
area itself (camera feed + AR glasses, already drawn by the caller)
and a few small live overlays (badges, hints, the CNN-busy pulse dot,
the face-analysis checklist) are touched every frame.

If the bundled TrueType font can't be loaded (missing/download failed/
corrupt), everything falls back to a simpler, purely cv2.putText-based
layout (_draw_ui_fallback, _show_gender_selection_fallback) so the app
never crashes for lack of a font.
"""
import cv2
import numpy as np
import os
import time
import math
import urllib.request
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from app.paths import ASSETS_DIR
from app.recommendations import (
    get_hair_rec, get_hair_rec_summary, get_grooming_rec, get_glasses_rec,
    get_glasses_rec_summary, get_color_rec, get_avoid_colors, get_color_swatch,
)
from app import feedback

# Supersampling factor for cached PIL images -- draw SS-times too big,
# downsample once with LANCZOS. Only ever applied to images that are
# cached and rebuilt rarely (see module docstring), never per frame.
SS = 3

# ──────────────────────────────────────────
# Layout
# ──────────────────────────────────────────
WINDOW_W, WINDOW_H = 1280, 720
PANEL_W      = 340
TOP_BAR_H    = 64
BOTTOM_TAB_H      = 44  # the row of 5 tabs
BOTTOM_TAGLINE_H  = 22  # "Better Style - Better You"
BOTTOM_BAR_H = BOTTOM_TAB_H + BOTTOM_TAGLINE_H
VIDEO_W = WINDOW_W - PANEL_W
VIDEO_H = WINDOW_H - TOP_BAR_H - BOTTOM_BAR_H

CARD_MARGIN = 14
CARD_GAP    = 10
CARD_RADIUS = 18

# Rounded-corner radius / border thickness for the live camera feed.
VIDEO_BORDER_RADIUS = 18
VIDEO_BORDER_THICK  = 3
VIDEO_HALO_THICK    = 2

# ──────────────────────────────────────────
# Colors (RGB, since PIL draws in RGB — converted to BGR once per
# cached image / constant when handed back to OpenCV)
# ──────────────────────────────────────────
BG_APP     = (10, 14, 26)      # #0A0E1A — app background
BG_CARD    = (15, 24, 48)      # #0F1830 — card background
BLUE_1     = (46, 158, 255)    # #2E9EFF
BLUE_2     = (27, 111, 224)    # #1B6FE0
PURPLE_1   = (168, 85, 247)    # #A855F7
PURPLE_2   = (224, 86, 253)    # #E056FD
TEXT_WHITE = (255, 255, 255)
TEXT_DIM   = (154, 163, 184)   # #9AA3B8
GREEN      = (34, 197, 94)     # #22C55E
AMBER      = (230, 172, 60)    # not in the reference palette, kept for low-confidence/warning text
RED        = (222, 90, 90)     # kept, unused by the (now gradient) confidence bar but available
PILL_BG    = (30, 35, 54)
RING_LIGHT = (176, 184, 204)   # thin light ring around outfit-color swatches

# Generic accent aliases: blue = face-shape/male-related, purple = skin-tone/female-related
BLUE_ACCENT   = BLUE_1
PURPLE_ACCENT = PURPLE_1

START_LETTERBOX_RGB = (3, 14, 28)  # #030E1C — background behind the start-screen image if its aspect ratio doesn't fill the window


def _mix(c1, c2, t):
    return tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))


BLUE_HALO_RGB   = _mix(BG_APP, BLUE_1, 0.45)     # faint version of the border color, for the halo ring
CARD_BG_BLEND   = _mix(BG_APP, BG_CARD, 0.88)    # flat approximation of a "slightly see-through" card over the navy background

BLUE_ACCENT_BGR    = BLUE_ACCENT[::-1]
PURPLE_ACCENT_BGR  = PURPLE_ACCENT[::-1]
BLUE_HALO_BGR       = BLUE_HALO_RGB[::-1]
AMBER_BGR    = AMBER[::-1]
GREEN_BGR    = GREEN[::-1]
BG_APP_BGR   = BG_APP[::-1]
START_LETTERBOX_BGR = START_LETTERBOX_RGB[::-1]


def _confidence_color(pct):
    if pct >= 60:
        return GREEN
    elif pct >= 40:
        return AMBER
    return RED


# ──────────────────────────────────────────
# Font loading: Poppins (Regular/Medium/SemiBold/Bold), downloaded on
# first run if missing. Every (style, size) actually used below is
# preloaded at both its normal size and its SS-times supersampled size
# -- see SS above -- so cached images can request the right glyph size
# directly instead of drawing small text into a 3x canvas.
# ──────────────────────────────────────────
FONT_DIR = str(ASSETS_DIR / 'fonts')
FONT_FILES = {
    'regular':  'Poppins-Regular.ttf',
    'medium':   'Poppins-Medium.ttf',
    'semibold': 'Poppins-SemiBold.ttf',
    'bold':     'Poppins-Bold.ttf',
}
FONT_URLS = {
    style: f'https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/{fname}'
    for style, fname in FONT_FILES.items()
}
FONT_SIZES = (11, 12, 13, 14, 16, 18, 20, 26, 28, 38)


def _download_fonts():
    try:
        os.makedirs(FONT_DIR, exist_ok=True)
        for style, fname in FONT_FILES.items():
            path = os.path.join(FONT_DIR, fname)
            if not os.path.exists(path):
                print(f"[ui] Downloading {fname}...")
                urllib.request.urlretrieve(FONT_URLS[style], path)
        print("[ui] Fonts ready!")
    except Exception as e:
        print(f"[ui] Couldn't download fonts ({e}); will use the simpler cv2 UI.")


class _FontDict(dict):
    """
    Dict of (style, size) -> ImageFont. If some code asks for a size
    that was never preloaded, falling through to a KeyError would
    crash render() on *every* frame -- and since render() runs inside
    main.py's per-frame try/except, that silently skips
    cv2.imshow()/waitKey() for the rest of the session too, freezing
    the window instead of just looking wrong. Falling back to the
    closest already-loaded size of the same style keeps the app
    running (a slightly-off font size is a cosmetic issue, not a
    crash) while still printing a one-time warning so the real fix --
    adding the size to FONT_SIZES -- doesn't get missed.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._warned = set()

    def __missing__(self, key):
        style, size = key
        candidates = [s for (st, s) in self.keys() if st == style]
        if not candidates:
            raise KeyError(key)  # this style was never loaded at all -- nothing to fall back to
        closest = min(candidates, key=lambda s: abs(s - size))
        if key not in self._warned:
            print(f"[ui] No preloaded font for {key}; using ('{style}', {closest}) instead. "
                  f"Add {size} to FONT_SIZES to fix this properly.")
            self._warned.add(key)
        return self[(style, closest)]


def _load_fonts():
    """Returns (fonts_dict, ok). fonts_dict maps (style, size) -> ImageFont, for size in FONT_SIZES and size*SS."""
    if not all(os.path.exists(os.path.join(FONT_DIR, f)) for f in FONT_FILES.values()):
        _download_fonts()
    if not all(os.path.exists(os.path.join(FONT_DIR, f)) for f in FONT_FILES.values()):
        return {}, False
    try:
        fonts = _FontDict()
        for style, fname in FONT_FILES.items():
            path = os.path.join(FONT_DIR, fname)
            for size in FONT_SIZES:
                fonts[(style, size)]      = ImageFont.truetype(path, size)
                fonts[(style, size * SS)] = ImageFont.truetype(path, size * SS)
        return fonts, True
    except Exception as e:
        print(f"[ui] Couldn't load UI fonts ({e}); will use the simpler cv2 UI.")
        return {}, False


# ──────────────────────────────────────────
# Low-level PIL drawing helpers. These all just draw into whatever
# canvas they're handed -- callers decide whether that canvas is a
# normal-resolution one-off (rare) or an SS-times supersampled cache
# (the common case), by scaling the coordinates/sizes they pass in.
# ──────────────────────────────────────────
def _wrap_text(draw, text, font, max_width):
    words = text.split(' ')
    lines, cur = [], ''
    for word in words:
        trial = (cur + ' ' + word).strip()
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _card(draw, x, y, w, h, radius, fill=CARD_BG_BLEND, border=None, border_width=1):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=fill)
    if border:
        draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, outline=border, width=border_width)


def _pill(draw, x, y, text, font, fg, bg, pad_x=10, pad_y=5):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    w, h = tw + pad_x * 2, th + pad_y * 2
    draw.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=bg)
    draw.text((x + pad_x - bbox[0], y + pad_y - bbox[1]), text, font=font, fill=fg)
    return w, h


def _gradient_image(w, h, color_a, color_b, axis='v'):
    """One-time gradient (vertical: a=top,b=bottom; horizontal: a=left,b=right) — never per frame."""
    a = np.array(color_a, dtype=np.float32)
    b = np.array(color_b, dtype=np.float32)
    if axis == 'v':
        t = np.linspace(0, 1, max(h, 1), dtype=np.float32)[:, None]
        col = (a[None, :] * (1 - t) + b[None, :] * t).astype(np.uint8)
        arr = np.repeat(col[:, None, :], max(w, 1), axis=1)
    else:
        t = np.linspace(0, 1, max(w, 1), dtype=np.float32)[:, None]
        row = (a[None, :] * (1 - t) + b[None, :] * t).astype(np.uint8)
        arr = np.repeat(row[None, :, :], max(h, 1), axis=0)
    return Image.fromarray(arr, 'RGB')


def _progress_bar(draw, x, y, w, h, frac, color, bg=(40, 45, 63)):
    """Flat-color pill progress bar (used where no gradient is called for)."""
    draw.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=bg)
    fw = int(w * max(0.0, min(frac, 1.0)))
    if fw > h:
        draw.rounded_rectangle([x, y, x + fw, y + h], radius=h // 2, fill=color)


def _progress_bar_gradient(img, x, y, w, h, frac, color_a, color_b, bg=(40, 45, 63)):
    """Pill progress bar filled with a smooth left-to-right gradient (the face-shape confidence bar)."""
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=bg)
    fw = int(w * max(0.0, min(frac, 1.0)))
    if fw > h:
        grad = _gradient_image(fw, h, color_a, color_b, axis='h')
        mask = Image.new('L', (fw, h), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, fw - 1, h - 1], radius=h // 2, fill=255)
        img.paste(grad, (int(x), int(y)), mask)


def _circle(draw, cx, cy, r, color, outline=None, outline_width=1):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=outline,
                 width=outline_width if outline else 0)


def _check_badge(draw, cx, cy, r, color=GREEN, line_width=2):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    draw.line([(cx - r * 0.5, cy), (cx - r * 0.1, cy + r * 0.4)], fill=(255, 255, 255), width=line_width)
    draw.line([(cx - r * 0.1, cy + r * 0.4), (cx + r * 0.55, cy - r * 0.35)], fill=(255, 255, 255), width=line_width)


def _empty_ring(draw, cx, cy, r, color, width=2):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=width)


def _icon_badge(img, cx, cy, r, color_a, color_b, icon_fn, icon_color):
    """Gradient-filled circle with an icon glyph on top — the "round icon with a soft gradient circle behind it" on each card."""
    size = int(2 * r) + 2
    grad = _gradient_image(size, size, color_a, color_b)
    mask = Image.new('L', (size, size), 0)
    ImageDraw.Draw(mask).ellipse([1, 1, size - 2, size - 2], fill=255)
    circle_img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    circle_img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(circle_img)
    icon_fn(d, size / 2, size / 2, r * 0.52, icon_color)
    img.paste(circle_img, (int(cx - size / 2), int(cy - size / 2)), circle_img)


def _draw_spaced_text(draw, cx, y, text, font, fill, spacing=3):
    """Centered text with extra gaps between letters, for the small-caps footer tagline."""
    widths = [draw.textlength(ch, font=font) for ch in text]
    total = sum(widths) + spacing * (len(text) - 1)
    x = cx - total / 2
    for ch, w_ in zip(text, widths):
        draw.text((x, y), ch, font=font, fill=fill)
        x += w_ + spacing


def _gradient_line(draw, x1, x2, y, color1, color2, thickness=2):
    """Short one-time horizontal gradient line (flanking the gender screen's title)."""
    n = max(int(abs(x2 - x1)), 1)
    for i in range(n):
        t = i / max(n - 1, 1)
        color = tuple(int(color1[c] * (1 - t) + color2[c] * t) for c in range(3))
        xi = x1 + (x2 - x1) * i / n
        draw.line([(xi, y), (xi + (x2 - x1) / n, y)], fill=color, width=thickness)


def _sparkle(draw, cx, cy, size, color):
    pts = []
    for i in range(8):
        ang = i * math.pi / 4
        r = size if i % 2 == 0 else size * 0.35
        pts.append((cx + r * math.sin(ang), cy - r * math.cos(ang)))
    draw.polygon(pts, fill=color)


def _male_silhouette(draw, cx, top_y, scale, color):
    r = 16 * scale
    draw.ellipse([cx - r, top_y, cx + r, top_y + 2 * r], fill=color)  # head
    body_top = top_y + 2 * r + 6 * scale
    body_w = 30 * scale
    draw.rounded_rectangle([cx - body_w, body_top, cx + body_w, body_top + 40 * scale],
                            radius=10 * scale, fill=color)  # shoulders/torso


def _female_silhouette(draw, cx, top_y, scale, color):
    r = 15 * scale
    draw.ellipse([cx - r, top_y, cx + r, top_y + 2 * r], fill=color)  # head
    body_top = top_y + 2 * r + 4 * scale
    draw.polygon([
        (cx - 10 * scale, body_top), (cx + 10 * scale, body_top),
        (cx + 32 * scale, body_top + 42 * scale), (cx - 32 * scale, body_top + 42 * scale),
    ], fill=color)  # dress/shoulders silhouette


def _arrow_button(draw, cx, cy, r, bg, fg):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=bg)
    draw.line([(cx - r * 0.22, cy - r * 0.38), (cx + r * 0.30, cy), (cx - r * 0.22, cy + r * 0.38)],
              fill=fg, width=3, joint='curve')


def _face_outline_icon(draw, cx, cy, r, color):
    draw.ellipse([cx - r, cy - r * 1.15, cx + r, cy + r * 1.15], outline=color, width=max(1, int(r * 0.14)))
    eye_r = max(1, r * 0.1)
    draw.ellipse([cx - r * 0.4 - eye_r, cy - eye_r, cx - r * 0.4 + eye_r, cy + eye_r], fill=color)
    draw.ellipse([cx + r * 0.4 - eye_r, cy - eye_r, cx + r * 0.4 + eye_r, cy + eye_r], fill=color)
    draw.arc([cx - r * 0.4, cy + r * 0.15, cx + r * 0.4, cy + r * 0.55], start=20, end=160,
              fill=color, width=max(1, int(r * 0.14)))


def _droplet_icon(draw, cx, cy, r, color):
    draw.ellipse([cx - r, cy - r * 0.2, cx + r, cy + r * 1.2], fill=color)
    draw.polygon([(cx, cy - r * 1.3), (cx - r * 0.6, cy), (cx + r * 0.6, cy)], fill=color)


def _scissors_icon(draw, cx, cy, r, color):
    """Hairstyle card icon: simple scissors -- two blade circles + crossed handles."""
    w = max(1, int(r * 0.16))
    draw.line([(cx - r * 0.7, cy - r * 0.6), (cx + r * 0.7, cy + r * 0.6)], fill=color, width=w)
    draw.line([(cx - r * 0.7, cy + r * 0.6), (cx + r * 0.7, cy - r * 0.6)], fill=color, width=w)
    ring_r = r * 0.28
    draw.ellipse([cx - r * 0.7 - ring_r, cy - r * 0.6 - ring_r, cx - r * 0.7 + ring_r, cy - r * 0.6 + ring_r],
                 outline=color, width=w)
    draw.ellipse([cx - r * 0.7 - ring_r, cy + r * 0.6 - ring_r, cx - r * 0.7 + ring_r, cy + r * 0.6 + ring_r],
                 outline=color, width=w)


def _spark_icon(draw, cx, cy, r, color):
    """Grooming card icon."""
    _sparkle(draw, cx, cy, r * 0.85, color)


def _palette_icon(draw, cx, cy, r, color):
    """Outfit colors card icon: three dots in a row."""
    dot_r = r * 0.34
    for dx in (-r * 0.62, 0, r * 0.62):
        draw.ellipse([cx + dx - dot_r, cy - dot_r, cx + dx + dot_r, cy + dot_r], fill=color)


def _glasses_icon(draw, cx, cy, r, color):
    lens_r = r * 0.42
    gap = r * 0.3
    w = max(1, int(r * 0.14))
    draw.ellipse([cx - gap - 2 * lens_r, cy - lens_r, cx - gap, cy + lens_r], outline=color, width=w)
    draw.ellipse([cx + gap, cy - lens_r, cx + gap + 2 * lens_r, cy + lens_r], outline=color, width=w)
    draw.line([(cx - gap, cy), (cx + gap, cy)], fill=color, width=w)


def _mic_icon(draw, cx, cy, size, color):
    w = size * 0.5
    lw = max(1, int(size * 0.14))
    draw.rounded_rectangle([cx - w / 2, cy - size, cx + w / 2, cy], radius=w / 2, fill=color)
    draw.arc([cx - size * 0.8, cy - size * 0.5, cx + size * 0.8, cy + size * 0.5],
              start=20, end=160, fill=color, width=lw)
    draw.line([(cx, cy + size * 0.5), (cx, cy + size * 0.8)], fill=color, width=lw)
    draw.line([(cx - size * 0.35, cy + size * 0.8), (cx + size * 0.35, cy + size * 0.8)], fill=color, width=lw)


def _gear_icon(draw, cx, cy, r, color):
    w = max(1, int(r * 0.18))
    draw.ellipse([cx - r * 0.55, cy - r * 0.55, cx + r * 0.55, cy + r * 0.55], outline=color, width=w)
    for i in range(8):
        ang = i * math.pi / 4
        x1, y1 = cx + r * 0.6 * math.cos(ang), cy + r * 0.6 * math.sin(ang)
        x2, y2 = cx + r * math.cos(ang), cy + r * math.sin(ang)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=w)


def _point_in_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def _downsample(img):
    """Final step for every SS-times-supersampled cached image: LANCZOS back down to real size."""
    w, h = img.size
    return img.resize((max(1, w // SS), max(1, h // SS)), Image.LANCZOS)


# ──────────────────────────────────────────
# Right panel cards. All coordinates/sizes here are already in SS
# (supersampled) space -- see _render_right_panel, which is the only
# caller and multiplies every dimension by SS before calling in.
# ──────────────────────────────────────────
def _draw_face_shape_card(img, draw, x, y, w, fonts, state, s):
    pad = 12 * s
    icon_r = 17 * s
    row_h = 2 * icon_r + 6 * s
    show_learned = (not state['analyzing']) and state['face_shape'] and state['learned_count'] > 0
    if not state['face_present']:
        height = row_h + pad + 10 * s
    elif state['analyzing']:
        height = row_h + 10 * s + 8 * s + 10 * s + (18 * s if state['near_edge'] else 0) + pad
    else:
        has_conf = state['face_shape'] and state['face_shape_confidence'] is not None
        height = row_h + (10 * s + 8 * s + 6 * s + 18 * s if has_conf else 0) + (18 * s if show_learned else 0) + pad

    _card(draw, x, y, w, height, radius=CARD_RADIUS * s, border=BLUE_ACCENT, border_width=max(1, int(1.4 * s)))

    icon_cx, icon_cy = x + pad + icon_r, y + pad + icon_r
    _icon_badge(img, icon_cx, icon_cy, icon_r, BLUE_1, BLUE_2, _face_outline_icon, TEXT_WHITE)

    text_x = x + pad + 2 * icon_r + 10 * s
    draw.text((text_x, y + pad), "FACE SHAPE", font=fonts[('semibold', 12 * s)], fill=BLUE_ACCENT)
    cy = y + pad + 15 * s

    if not state['face_present']:
        draw.text((text_x, cy), "No face detected", font=fonts[('medium', 16 * s)], fill=TEXT_DIM)
    elif state['analyzing']:
        draw.text((text_x, cy), "Analyzing...", font=fonts[('medium', 16 * s)], fill=BLUE_ACCENT)
        cy = y + row_h + pad
        collected, total = state['analysis_progress']
        _progress_bar(draw, x + pad, cy, w - 2 * pad, 8 * s, collected / max(total, 1), BLUE_ACCENT)
        cy += 8 * s + 10 * s
        if state['near_edge']:
            draw.text((x + pad, cy), "Move to center", font=fonts[('regular', 12 * s)], fill=AMBER)
    else:
        label = state['face_shape'] or "Detecting..."
        draw.text((text_x, cy), label, font=fonts[('bold', 20 * s)],
                  fill=TEXT_WHITE if state['face_shape'] else TEXT_DIM)
        if state['face_shape'] and state['face_shape_confidence'] is not None:
            label_w = draw.textlength(label, font=fonts[('bold', 20 * s)])
            _check_badge(draw, text_x + label_w + 16 * s, cy + 11 * s, r=8 * s)
        cy = y + row_h + pad
        conf = state['face_shape_confidence']
        if state['face_shape'] and conf is not None:
            # Smooth purple-to-blue gradient pill, per the reference design
            # (this deliberately replaces the old red/amber/green confidence
            # coloring with a fixed brand gradient -- the fill WIDTH still
            # tracks the confidence percentage, only the color styling changed).
            _progress_bar_gradient(img, x + pad, cy, w - 2 * pad, 8 * s, conf / 100, PURPLE_1, BLUE_1)
            cy += 8 * s + 6 * s
            draw.text((x + pad, cy), f"{conf:.0f}% confidence (locked)",
                      font=fonts[('regular', 12 * s)], fill=TEXT_DIM)
            cy += 18 * s
        if show_learned:
            draw.text((x + pad, cy), f"Learned from {state['learned_count']} ratings",
                      font=fonts[('regular', 12 * s)], fill=BLUE_ACCENT)

    return y + height + CARD_GAP * s


def _draw_skin_tone_card(img, draw, x, y, w, fonts, state, s):
    pad = 12 * s
    icon_r = 17 * s
    row_h = 2 * icon_r + 6 * s
    show_hint = bool(state['skin_tone']) or state.get('skin_tone_low_confidence')
    height = row_h + pad + (16 * s if state.get('skin_tone_low_confidence') else 0) + (18 * s if show_hint else 6 * s)

    _card(draw, x, y, w, height, radius=CARD_RADIUS * s, border=PURPLE_ACCENT, border_width=max(1, int(1.4 * s)))

    icon_cx, icon_cy = x + pad + icon_r, y + pad + icon_r
    _icon_badge(img, icon_cx, icon_cy, icon_r, PURPLE_1, PURPLE_2, _droplet_icon, TEXT_WHITE)

    text_x = x + pad + 2 * icon_r + 10 * s
    draw.text((text_x, y + pad), "SKIN TONE", font=fonts[('semibold', 12 * s)], fill=PURPLE_ACCENT)
    cy = y + pad + 15 * s
    if state['skin_tone']:
        draw.text((text_x, cy), state['skin_tone'], font=fonts[('bold', 20 * s)], fill=TEXT_WHITE)
        swatch = state['skin_swatch_rgb'] or (150, 150, 150)
        sw_r = 9 * s
        _circle(draw, text_x + draw.textlength(state['skin_tone'], font=fonts[('bold', 20 * s)]) + 16 * s,
                cy + 11 * s, sw_r, swatch, outline=RING_LIGHT, outline_width=max(1, int(1.2 * s)))
        cy = y + row_h + pad
        if state.get('skin_tone_low_confidence'):
            draw.text((x + pad, cy), "Rough estimate — shadow detected", font=fonts[('regular', 12 * s)], fill=AMBER)
            cy += 16 * s
    else:
        draw.text((text_x, cy), "Detecting...", font=fonts[('medium', 16 * s)], fill=TEXT_DIM)
        cy = y + row_h + pad
    if show_hint:
        draw.text((x + pad, cy), "For best results, remove hats and ensure even lighting",
                  font=fonts[('regular', 11 * s)], fill=TEXT_DIM)
    return y + height + CARD_GAP * s


def _card_header(img, draw, x, y, w, fonts, s, title, icon_fn, color_a, color_b, icon_color=TEXT_WHITE):
    """Shared icon+uppercase-title header row used by the list-style cards (hairstyle/grooming/outfit/glasses/voice)."""
    pad = 12 * s
    icon_r = 14 * s
    icon_cx, icon_cy = x + pad + icon_r, y + pad + icon_r
    _icon_badge(img, icon_cx, icon_cy, icon_r, color_a, color_b, icon_fn, icon_color)
    text_x = x + pad + 2 * icon_r + 10 * s
    draw.text((text_x, y + pad + icon_r - 8 * s), title, font=fonts[('semibold', 12 * s)], fill=BLUE_ACCENT)
    return y + pad + 2 * icon_r + 6 * s  # y for the content that follows the header


def _draw_hairstyle_card(img, draw, x, y, w, fonts, state, s):
    pad = 12 * s
    content_w = w - 2 * pad
    recs = state['hair_recs'][:3]
    items = []
    header_h = pad + 2 * 14 * s + 6 * s
    height = header_h
    for rec in recs:
        why_lines = _wrap_text(draw, rec['why'], fonts[('regular', 12 * s)], content_w - 4 * s)
        item_h = 20 * s + len(why_lines) * 15 * s + 6 * s
        items.append((rec, why_lines))
        height += item_h
    if not recs:
        height += 20 * s
    height += pad

    _card(draw, x, y, w, height, radius=CARD_RADIUS * s, border=BLUE_ACCENT, border_width=max(1, int(1.4 * s)))
    cy = _card_header(img, draw, x, y, w, fonts, s, "HAIRSTYLE", _scissors_icon, BLUE_1, BLUE_2)
    if not recs:
        draw.text((x + pad, cy), "---", font=fonts[('regular', 14 * s)], fill=TEXT_DIM)
    for rec, why_lines in items:
        name_color = TEXT_DIM if rec.get('less_popular') else TEXT_WHITE
        draw.text((x + pad, cy), rec['name'], font=fonts[('medium', 14 * s)], fill=name_color)
        if rec.get('less_popular'):
            name_w = draw.textlength(rec['name'], font=fonts[('medium', 14 * s)])
            draw.text((x + pad + name_w + 8 * s, cy + 2 * s), "less popular", font=fonts[('regular', 12 * s)], fill=AMBER)
        cy += 20 * s
        for line in why_lines:
            draw.text((x + pad + 4 * s, cy), line, font=fonts[('regular', 12 * s)], fill=TEXT_DIM)
            cy += 15 * s
        cy += 6 * s
    return y + height + CARD_GAP * s


def _draw_grooming_card(img, draw, x, y, w, fonts, state, s):
    pad = 12 * s
    content_w = w - 2 * pad - 12 * s
    tips = state['grooming_tips']
    tip_lines = []
    header_h = pad + 2 * 14 * s + 4 * s
    height = header_h
    for tip in tips:
        wrapped = _wrap_text(draw, tip, fonts[('regular', 14 * s)], content_w)
        tip_lines.append(wrapped)
        height += len(wrapped) * 19 * s + 5 * s
    if not tips:
        height += 18 * s
    height += pad

    _card(draw, x, y, w, height, radius=CARD_RADIUS * s, border=BLUE_ACCENT, border_width=max(1, int(1.4 * s)))
    cy = _card_header(img, draw, x, y, w, fonts, s, "GROOMING TIPS", _spark_icon, BLUE_1, BLUE_2)
    if not tips:
        draw.text((x + pad, cy), "---", font=fonts[('regular', 14 * s)], fill=TEXT_DIM)
    for wrapped in tip_lines:
        for i, line in enumerate(wrapped):
            prefix = "• " if i == 0 else "   "
            draw.text((x + pad, cy), prefix + line, font=fonts[('regular', 14 * s)], fill=TEXT_WHITE)
            cy += 19 * s
        cy += 5 * s
    return y + height + CARD_GAP * s


def _wrap_color_names(draw, names, font, max_w, dim_label=None, r=7):
    """Groups color names into rows that fit within max_w — wraps rather than dropping overflow items."""
    lines, current, cx = [], [], (draw.textlength(dim_label, font=font) + 6 if dim_label else 0)
    for name in names:
        item_w = 2 * r + 6 + draw.textlength(name, font=font) + 14
        if cx + item_w > max_w and current:
            lines.append(current)
            current, cx = [], 0
        current.append(name)
        cx += item_w
    if current:
        lines.append(current)
    return lines or [[]]


def _draw_color_lines(draw, x, y, lines, font, name_color, dim_label=None, row_h=20, r=7, outline_w=1):
    for i, line_names in enumerate(lines):
        cx = x
        if i == 0 and dim_label:
            draw.text((cx, y), dim_label, font=font, fill=TEXT_DIM)
            cx += draw.textlength(dim_label, font=font) + 6
        for name in line_names:
            rgb = get_color_swatch(name)
            _circle(draw, cx + r, y + r + 1, r, rgb, outline=RING_LIGHT, outline_width=outline_w)
            cx += 2 * r + 6
            draw.text((cx, y), name, font=font, fill=name_color)
            cx += draw.textlength(name, font=font) + 14
        y += row_h
    return y


def _draw_outfit_card(img, draw, x, y, w, fonts, state, s):
    pad = 12 * s
    content_w = w - 2 * pad
    font = fonts[('regular', 12 * s)]
    r = 7 * s
    rec_names   = [c.strip() for c in state['color_rec'].split(',')] if state['color_rec'] != '---' else []
    avoid_names = [c.strip() for c in state['avoid_colors'].split(',')] if state['avoid_colors'] != '---' else []

    wear_lines  = _wrap_color_names(draw, rec_names, font, content_w, r=r)
    avoid_lines = _wrap_color_names(draw, avoid_names, font, content_w, dim_label="Avoid:", r=r) if avoid_names else []

    header_h = pad + 2 * 14 * s + 4 * s
    height = header_h + len(wear_lines) * 20 * s + len(avoid_lines) * 20 * s + pad
    _card(draw, x, y, w, height, radius=CARD_RADIUS * s, border=BLUE_ACCENT, border_width=max(1, int(1.4 * s)))
    cy = _card_header(img, draw, x, y, w, fonts, s, "OUTFIT COLORS", _palette_icon, BLUE_1, BLUE_2)
    cy = _draw_color_lines(draw, x + pad, cy, wear_lines, font, TEXT_WHITE, row_h=20 * s, r=r, outline_w=max(1, int(1.2 * s)))
    if avoid_lines:
        _draw_color_lines(draw, x + pad, cy, avoid_lines, font, TEXT_DIM, dim_label="Avoid:", row_h=20 * s, r=r, outline_w=max(1, int(1.2 * s)))
    return y + height + CARD_GAP * s


def _draw_glasses_card(img, draw, x, y, w, fonts, state, s):
    pad = 12 * s
    content_w = w - 2 * pad
    recs = state['glasses_types'][:3]
    items = []
    header_h = pad + 2 * 14 * s + 4 * s
    height = header_h
    for rec in recs:
        why_lines = _wrap_text(draw, rec['why'], fonts[('regular', 12 * s)], content_w - 4 * s)
        items.append((rec, why_lines))
        height += 18 * s + len(why_lines) * 15 * s + 5 * s
    if not recs:
        height += 18 * s
    info_line = bool(state['glasses_names'])
    height += (18 * s if info_line else 0) + pad

    _card(draw, x, y, w, height, radius=CARD_RADIUS * s, border=BLUE_ACCENT, border_width=max(1, int(1.4 * s)))

    if state['is_recommended']:
        badge_text = "REC"
        bw = draw.textlength(badge_text, font=fonts[('bold', 12 * s)]) + 14 * s
        draw.rounded_rectangle([x + w - pad - bw, y + 8 * s, x + w - pad, y + 28 * s], radius=10 * s, fill=GREEN)
        draw.text((x + w - pad - bw + 7 * s, y + 11 * s), badge_text, font=fonts[('bold', 12 * s)], fill=(15, 25, 18))

    cy = _card_header(img, draw, x, y, w, fonts, s, "GLASSES", _glasses_icon, BLUE_1, BLUE_2)
    if not recs:
        draw.text((x + pad, cy), "---", font=fonts[('regular', 14 * s)], fill=TEXT_DIM)
    for rec, why_lines in items:
        name_color = TEXT_DIM if rec.get('less_popular') else TEXT_WHITE
        draw.text((x + pad, cy), rec['name'], font=fonts[('medium', 14 * s)], fill=name_color)
        if rec.get('less_popular'):
            name_w = draw.textlength(rec['name'], font=fonts[('medium', 14 * s)])
            draw.text((x + pad + name_w + 8 * s, cy + 2 * s), "less popular", font=fonts[('regular', 12 * s)], fill=AMBER)
        cy += 18 * s
        for line in why_lines:
            draw.text((x + pad + 4 * s, cy), line, font=fonts[('regular', 12 * s)], fill=TEXT_DIM)
            cy += 15 * s
        cy += 5 * s
    if info_line:
        info = f"{state['glasses_idx'] + 1}/{state['total_glasses']}  {state['glasses_current_name']}"
        draw.text((x + pad, cy), info, font=fonts[('regular', 12 * s)], fill=TEXT_DIM)
    return y + height + CARD_GAP * s


def _draw_voice_card(img, draw, x, y, w, fonts, state, s):
    pad = 12 * s
    mic_available = state.get('mic_available', True)
    last_text = state.get('last_voice_text')
    status = state['voice_status']

    if not mic_available:
        status_text, status_color = "No microphone detected", TEXT_DIM
    elif status == 'listening':
        status_text, status_color = "Listening...", GREEN
    elif status == 'on':
        status_text, status_color = "On — say a command", GREEN
    else:
        status_text, status_color = "Ready — press V to enable", TEXT_DIM

    commands = ['"face shape" / "skin tone"', '"glasses" / "next" / "previous"',
                '"hairstyle" / "outfit"', '"help" / "quit"']

    header_h = pad + 2 * 14 * s + 4 * s
    height = header_h + 22 * s + (18 * s if last_text else 0) + 15 * s + 17 * s + len(commands) * 17 * s + pad
    _card(draw, x, y, w, height, radius=CARD_RADIUS * s, border=BLUE_ACCENT, border_width=max(1, int(1.4 * s)))
    cy = _card_header(img, draw, x, y, w, fonts, s, "VOICE ASSISTANT", _mic_icon_wrapper, BLUE_1, BLUE_2)
    draw.text((x + pad, cy), status_text, font=fonts[('medium', 16 * s)], fill=status_color)
    cy += 22 * s
    if last_text:
        draw.text((x + pad, cy), f'Heard: "{last_text}"', font=fonts[('regular', 12 * s)], fill=TEXT_DIM)
        cy += 18 * s
    cy += 15 * s
    draw.text((x + pad, cy), "Try saying:", font=fonts[('regular', 12 * s)], fill=TEXT_DIM)
    cy += 17 * s
    for c in commands:
        draw.text((x + pad + 4 * s, cy), c, font=fonts[('regular', 12 * s)], fill=TEXT_WHITE)
        cy += 17 * s
    return y + height + CARD_GAP * s


def _mic_icon_wrapper(draw, cx, cy, r, color):
    _mic_icon(draw, cx, cy, r, color)


# ──────────────────────────────────────────
# Bottom tab bar: which tab is active decides which card(s) the right
# panel shows (see _render_right_panel below).
# ──────────────────────────────────────────
TAB_KEYS = ['face', 'hairstyle', 'glasses', 'outfit', 'voice']
TAB_LABELS = {
    'face': 'Face Analysis', 'hairstyle': 'Hairstyle', 'glasses': 'Glasses',
    'outfit': 'Outfit Colors', 'voice': 'Voice Assistant',
}


def hit_test_tab(x, y):
    """Returns the tab key clicked at window coords (x, y), or None if the click wasn't on the tab row."""
    tab_top = WINDOW_H - BOTTOM_BAR_H
    tab_bottom = tab_top + BOTTOM_TAB_H
    if not (tab_top <= y < tab_bottom):
        return None
    idx = int(x // (WINDOW_W / len(TAB_KEYS)))
    if 0 <= idx < len(TAB_KEYS):
        return TAB_KEYS[idx]
    return None


def next_tab(current, step=1):
    """Cycles to the next/previous tab — used by the '[' / ']' keyboard shortcut."""
    i = TAB_KEYS.index(current) if current in TAB_KEYS else 0
    return TAB_KEYS[(i + step) % len(TAB_KEYS)]


def _render_right_panel(fonts, state):
    s = SS
    x, w = CARD_MARGIN * s, (PANEL_W - 2 * CARD_MARGIN) * s
    # Two passes aren't needed: each card function measures its own
    # content height before drawing, so we just need a canvas tall
    # enough for the worst case (comfortably above it — this is just
    # an initial buffer size, not the final image; it gets cropped to
    # the real content height below, and the caller scales the result
    # down to fit the visible panel area if it's still too tall).
    canvas_h = 1400 * s
    img = Image.new('RGB', (PANEL_W * s, canvas_h), BG_APP)
    draw = ImageDraw.Draw(img)
    y = CARD_MARGIN * s
    tab = state.get('active_tab', 'face')
    if tab == 'face':
        y = _draw_face_shape_card(img, draw, x, y, w, fonts, state, s)
        y = _draw_skin_tone_card(img, draw, x, y, w, fonts, state, s)
    elif tab == 'hairstyle':
        y = _draw_hairstyle_card(img, draw, x, y, w, fonts, state, s)
        y = _draw_grooming_card(img, draw, x, y, w, fonts, state, s)
    elif tab == 'glasses':
        y = _draw_glasses_card(img, draw, x, y, w, fonts, state, s)
    elif tab == 'outfit':
        y = _draw_outfit_card(img, draw, x, y, w, fonts, state, s)
    elif tab == 'voice':
        y = _draw_voice_card(img, draw, x, y, w, fonts, state, s)
    img = img.crop((0, 0, PANEL_W * s, min(y, canvas_h)))
    img = _downsample(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _render_top_bar(fonts, state):
    s = SS
    img = Image.new('RGB', (WINDOW_W * s, TOP_BAR_H * s), BG_APP)
    draw = ImageDraw.Draw(img)

    # App icon + title
    _sparkle(draw, 26 * s, TOP_BAR_H * s // 2, 9 * s, BLUE_1)
    title = "AI Grooming Assistant"
    title_font = fonts[('bold', 18 * s)]
    draw.text((44 * s, TOP_BAR_H * s // 2 - 12 * s), title, font=title_font, fill=TEXT_WHITE)
    title_end_x = 44 * s + draw.textlength(title, font=title_font)

    if state.get('rule_based_mode'):
        # Kept visible on screen (not just printed to the console) --
        # stress test case 7 needs a visible badge when the trained
        # model files are missing and the app falls back to the
        # geometric rule-based classifier.
        _pill(draw, title_end_x + 16 * s, TOP_BAR_H * s // 2 - 11 * s, "Rule-based mode",
              fonts[('medium', 11 * s)], AMBER, PILL_BG)

    # Gender pill-style toggle, centered. Display-only: gender is fixed
    # for the session once chosen on the selection screen, so this
    # isn't a live switch -- it just shows which side is active,
    # colored to match the male/female accent.
    is_male = state['gender'] == 'male'
    seg_w, seg_h = 56 * s, 26 * s
    px = WINDOW_W * s // 2 - seg_w
    py = (TOP_BAR_H * s - seg_h) // 2
    draw.rounded_rectangle([px, py, px + 2 * seg_w, py + seg_h], radius=seg_h // 2, fill=PILL_BG)
    active_x = px if is_male else px + seg_w
    draw.rounded_rectangle([active_x, py, active_x + seg_w, py + seg_h], radius=seg_h // 2,
                            fill=BLUE_ACCENT if is_male else PURPLE_ACCENT)
    mf = fonts[('medium', 11 * s)]
    draw.text((px + seg_w / 2 - draw.textlength("MALE", font=mf) / 2, py + 7 * s), "MALE",
              font=mf, fill=TEXT_WHITE if is_male else TEXT_DIM)
    draw.text((px + seg_w * 1.5 - draw.textlength("FEMALE", font=mf) / 2, py + 7 * s), "FEMALE",
              font=mf, fill=TEXT_WHITE if not is_male else TEXT_DIM)

    # Voice status + mic icon, then the settings gear -- right side.
    mic_available = state.get('mic_available', True)
    status = state['voice_status']
    if not mic_available:
        vtext, vcolor = "Voice Assistant: Unavailable", TEXT_DIM
    elif status == 'listening':
        vtext, vcolor = "Voice Assistant: Listening", GREEN
    elif status == 'on':
        vtext, vcolor = "Voice Assistant: On", GREEN
    else:
        vtext, vcolor = "Voice Assistant: Ready", TEXT_DIM

    vf = fonts[('regular', 12 * s)]
    vw = draw.textlength(vtext, font=vf)
    gear_cx = WINDOW_W * s - 24 * s
    vx = gear_cx - 22 * s - vw
    _mic_icon(draw, vx - 14 * s, TOP_BAR_H * s // 2, 8 * s, vcolor)
    draw.text((vx, TOP_BAR_H * s // 2 - 7 * s), vtext, font=vf, fill=vcolor)
    _gear_icon(draw, gear_cx, TOP_BAR_H * s // 2, 11 * s, TEXT_DIM)  # decorative, no settings menu exists

    img = _downsample(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _render_bottom_bar(fonts, state):
    s = SS
    img = Image.new('RGB', (WINDOW_W * s, BOTTOM_BAR_H * s), BG_APP)
    draw = ImageDraw.Draw(img)

    active = state.get('active_tab', 'face')
    seg_w = WINDOW_W * s / len(TAB_KEYS)
    tf_active = fonts[('semibold', 13 * s)]
    tf        = fonts[('regular', 13 * s)]
    for i, key in enumerate(TAB_KEYS):
        cx = int(seg_w * i + seg_w / 2)
        label = TAB_LABELS[key]
        is_active = key == active
        color = BLUE_ACCENT if is_active else TEXT_DIM
        font = tf_active if is_active else tf
        lw = draw.textlength(label, font=font)
        draw.text((cx - lw / 2, 12 * s), label, font=font, fill=color)
        if is_active:
            draw.rounded_rectangle([cx - lw / 2 - 4 * s, 32 * s, cx + lw / 2 + 4 * s, 34 * s],
                                    radius=1 * s, fill=BLUE_ACCENT)

    draw.line([(0, BOTTOM_TAB_H * s), (WINDOW_W * s, BOTTOM_TAB_H * s)], fill=BG_CARD, width=max(1, s))

    tag_font = fonts[('regular', 11 * s)]
    tag = "Better Style - Better You"
    tw = draw.textlength(tag, font=tag_font)
    draw.text((WINDOW_W * s / 2 - tw / 2, (BOTTOM_TAB_H + 5) * s), tag, font=tag_font, fill=TEXT_DIM)

    img = _downsample(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _fit_frame(frame, target_w, target_h):
    """Resizes frame to fit inside target_w x target_h, preserving aspect ratio, letterboxed in black."""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(frame, (nw, nh), interpolation=interp)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off, y_off = (target_w - nw) // 2, (target_h - nh) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    return canvas


def _corner_cut_mask(radius):
    """radius x radius boolean mask, True where a TOP-LEFT rounded corner should be painted background."""
    img = Image.new('L', (radius, radius), 0)
    ImageDraw.Draw(img).pieslice([0, 0, 2 * radius, 2 * radius], 180, 270, fill=255)
    keep = np.array(img) > 0
    return ~keep


def _render_badge(text, color, font):
    """Small solid pill badge ('Live Analysis' / 'Camera Active'), supersampled+cached once and reused every frame."""
    s = SS
    tmp = Image.new('RGB', (1, 1))
    bbox = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 8 * s, 4 * s
    w, h = tw + pad_x * 2, th + pad_y * 2
    img = Image.new('RGB', (int(w), int(h)), color)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=h // 2, fill=color)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=(10, 14, 26))
    img = _downsample(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _render_checklist(fonts, face_ready, skin_ready, style_ready):
    """
    Small floating 'Face Analysis' checklist for the top-left of the
    camera view. Ticks green once each result is actually ready:
    Face Shape (locked), Skin Tone (detected), and Style Preferences
    (at least one rating has been given, so recommendations are being
    personalized) -- deliberately no "Hair Analysis" row, since hair
    itself is never detected, only recommended.
    """
    s = SS
    pad = 10 * s
    row_h = 20 * s
    rows = [("Face Shape", face_ready), ("Skin Tone", skin_ready), ("Style Preferences", style_ready)]
    font = fonts[('medium', 12 * s)]
    title_font = fonts[('semibold', 12 * s)]
    max_text_w = max(ImageDraw.Draw(Image.new('RGB', (1, 1))).textlength(t, font=font) for t, _ in rows)
    w = int(pad * 2 + 20 * s + max_text_w)
    h = int(pad * 2 + 18 * s + row_h * len(rows))

    img = Image.new('RGB', (w, h), CARD_BG_BLEND)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=CARD_RADIUS * s // 2, outline=BLUE_ACCENT,
                            width=max(1, int(1.4 * s)))
    draw.text((pad, pad), "FACE ANALYSIS", font=title_font, fill=BLUE_ACCENT)
    cy = pad + 18 * s
    for label, ready in rows:
        cx, ccy = pad + 7 * s, cy + row_h / 2
        if ready:
            _check_badge(draw, cx, ccy, r=7 * s)
        else:
            _empty_ring(draw, cx, ccy, r=7 * s, color=(80, 88, 108), width=max(1, int(1.2 * s)))
        draw.text((pad + 20 * s, cy + 3 * s), label, font=font, fill=TEXT_WHITE if ready else TEXT_DIM)
        cy += row_h

    img = _downsample(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _panel_cache_key(state):
    hair = tuple((r['name'], r['why'], r.get('less_popular', False)) for r in state['hair_recs'])
    swatch = tuple(state['skin_swatch_rgb']) if state['skin_swatch_rgb'] is not None else None
    conf = None if state['face_shape_confidence'] is None else round(state['face_shape_confidence'])
    glasses_types = tuple((r['name'], r['why'], r.get('less_popular', False)) for r in state['glasses_types'])
    return (
        state.get('active_tab'),
        state['face_shape'], conf, state['analyzing'], state['analysis_progress'],
        state['near_edge'], state['face_present'], state['skin_tone'], swatch, hair,
        state['skin_tone_low_confidence'],
        tuple(state['grooming_tips']), state['color_rec'], state['avoid_colors'],
        glasses_types, state['glasses_idx'], state['glasses_current_name'],
        state['total_glasses'], state['is_recommended'], state['learned_count'],
        state.get('voice_status'), state.get('mic_available'), state.get('last_voice_text'),
    )


def _checklist_key(state):
    face_ready = bool(state['face_shape']) and not state['analyzing']
    skin_ready = bool(state['skin_tone'])
    style_ready = state.get('learned_count', 0) > 0
    return (face_ready, skin_ready, style_ready)


class UIRenderer:
    """
    Owns the UI font(s), the precomputed video-border corner masks/
    badges, and the cached top/right/bottom images. Create one
    instance before the camera loop and call render() every frame.
    """
    def __init__(self):
        self.fonts, self.pil_ok = _load_fonts()
        self._top_key,    self._top_img    = None, None
        self._panel_key,  self._panel_img  = None, None
        self._bottom_key, self._bottom_img = None, None
        self._checklist_key_val, self._checklist_img = None, None

        r = VIDEO_BORDER_RADIUS
        cut = _corner_cut_mask(r)
        self._tl_mask = cut
        self._tr_mask = np.fliplr(cut)
        self._bl_mask = np.flipud(cut)
        self._br_mask = np.fliplr(np.flipud(cut))

        if self.pil_ok:
            badge_font = self.fonts[('semibold', 11 * SS)]
            self._live_badge_img   = _render_badge("Live Analysis", GREEN, badge_font)
            self._camera_badge_img = _render_badge("Camera Active", BLUE_ACCENT, badge_font)
        else:
            self._live_badge_img = self._camera_badge_img = None

    def _draw_video_border(self, video):
        r = VIDEO_BORDER_RADIUS
        t = VIDEO_BORDER_THICK
        w, h = VIDEO_W, VIDEO_H

        # Cheap true rounded corners: only the 4 small (r x r) corner
        # squares are touched, not the whole frame (a full-frame mask
        # measured ~3.7ms/frame in testing -- these 4 tiny slices cost
        # microseconds).
        video[0:r, 0:r][self._tl_mask] = BG_APP_BGR
        video[0:r, w - r:w][self._tr_mask] = BG_APP_BGR
        video[h - r:h, 0:r][self._bl_mask] = BG_APP_BGR
        video[h - r:h, w - r:w][self._br_mask] = BG_APP_BGR

        def ring(color, radius, thick):
            cv2.line(video, (radius, thick // 2), (w - radius, thick // 2), color, thick, cv2.LINE_AA)
            cv2.line(video, (radius, h - 1 - thick // 2), (w - radius, h - 1 - thick // 2), color, thick, cv2.LINE_AA)
            cv2.line(video, (thick // 2, radius), (thick // 2, h - radius), color, thick, cv2.LINE_AA)
            cv2.line(video, (w - 1 - thick // 2, radius), (w - 1 - thick // 2, h - radius), color, thick, cv2.LINE_AA)
            cv2.ellipse(video, (radius, radius), (radius, radius), 0, 180, 270, color, thick, cv2.LINE_AA)
            cv2.ellipse(video, (w - 1 - radius, radius), (radius, radius), 0, 270, 360, color, thick, cv2.LINE_AA)
            cv2.ellipse(video, (radius, h - 1 - radius), (radius, radius), 0, 90, 180, color, thick, cv2.LINE_AA)
            cv2.ellipse(video, (w - 1 - radius, h - 1 - radius), (radius, radius), 0, 0, 90, color, thick, cv2.LINE_AA)

        # Faint halo just inside the solid border -- a real blurred
        # glow would cost several ms every frame for a border that
        # never moves, so this flat lighter-color ring approximates it.
        ring(BLUE_HALO_BGR, r - t, VIDEO_HALO_THICK)
        ring(BLUE_ACCENT_BGR, r, t)

    def render(self, video_frame_bgr, state):
        if not self.pil_ok:
            return _draw_ui_fallback(video_frame_bgr, state)

        # np.empty + only filling the (small, rare) leftover gap below
        # is far cheaper than filling the whole 1280x720 canvas every
        # frame (measured ~6ms for a full-canvas tuple-broadcast fill
        # vs ~0.1ms for a small gap) — video/top/panel/bottom already
        # cover the entire canvas except for that possible gap.
        canvas = np.empty((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)

        video = _fit_frame(video_frame_bgr, VIDEO_W, VIDEO_H)
        self._draw_video_border(video)

        if self._live_badge_img is not None:
            bh, bw = self._live_badge_img.shape[:2]
            video[10:10 + bh, 10:10 + bw] = self._live_badge_img
        if self._camera_badge_img is not None:
            bh, bw = self._camera_badge_img.shape[:2]
            video[VIDEO_H - 10 - bh:VIDEO_H - 10, 10:10 + bw] = self._camera_badge_img

        if self.pil_ok:
            checklist_key = _checklist_key(state)
            if checklist_key != self._checklist_key_val:
                self._checklist_img = _render_checklist(self.fonts, *checklist_key)
                self._checklist_key_val = checklist_key
            ch, cw = self._checklist_img.shape[:2]
            cly = 10 + (self._live_badge_img.shape[0] if self._live_badge_img is not None else 0) + 8
            video[cly:cly + ch, 10:10 + cw] = self._checklist_img

        if state.get('multiple_faces'):
            # Centered at the top rather than left-aligned -- the
            # top-left corner is taken by the Face Analysis checklist.
            text = "Multiple faces detected - using the largest"
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(video, text, ((VIDEO_W - tw) // 2, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, AMBER_BGR, 2, cv2.LINE_AA)
        if state['analyzing'] and state['near_edge']:
            cv2.putText(video, "Move to center", (36, VIDEO_H - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, AMBER_BGR, 2, cv2.LINE_AA)
        if state['cnn_busy']:
            pulse = (math.sin(time.time() * 6) + 1) / 2
            radius = int(4 + pulse * 3)
            cv2.circle(video, (VIDEO_W - 40, 40), radius, BLUE_ACCENT_BGR, -1, cv2.LINE_AA)
        canvas[TOP_BAR_H:TOP_BAR_H + VIDEO_H, 0:VIDEO_W] = video

        top_key = (state['gender'], state['voice_status'], state.get('mic_available', True),
                   state.get('rule_based_mode', False))
        if top_key != self._top_key:
            self._top_img = _render_top_bar(self.fonts, state)
            self._top_key = top_key
        canvas[0:TOP_BAR_H, 0:WINDOW_W] = self._top_img

        panel_key = _panel_cache_key(state)
        if panel_key != self._panel_key:
            panel_img = _render_right_panel(self.fonts, state)
            avail_h = WINDOW_H - TOP_BAR_H
            # Cards size themselves to their content and can occasionally
            # add up to more than the visible panel height (e.g. long
            # wrapped hairstyle text) — rather than truncating the last
            # card, scale the whole panel down slightly to fit. Only
            # happens when the cache is rebuilt, not every frame.
            if panel_img.shape[0] > avail_h:
                panel_img = cv2.resize(panel_img, (PANEL_W, avail_h), interpolation=cv2.INTER_AREA)
            self._panel_img = panel_img
            self._panel_key = panel_key
        h = self._panel_img.shape[0]
        canvas[TOP_BAR_H:TOP_BAR_H + h, WINDOW_W - PANEL_W:WINDOW_W] = self._panel_img
        if TOP_BAR_H + h < WINDOW_H:
            canvas[TOP_BAR_H + h:WINDOW_H, WINDOW_W - PANEL_W:WINDOW_W] = BG_APP_BGR

        bottom_key = state.get('active_tab')
        if bottom_key != self._bottom_key:
            self._bottom_img = _render_bottom_bar(self.fonts, state)
            self._bottom_key = bottom_key
        canvas[WINDOW_H - BOTTOM_BAR_H:WINDOW_H, 0:WINDOW_W] = self._bottom_img

        return canvas


def build_ui_state(gender, voice_status, fps, face_shape, face_shape_confidence,
                    analyzing, analysis_progress, near_edge, cnn_busy,
                    skin_tone, skin_swatch_rgb,
                    glasses_idx, glasses_names, suggested, total_glasses,
                    face_present=True, multiple_faces=False, rule_based_mode=False,
                    skin_tone_low_confidence=False, active_tab='face',
                    mic_available=True, last_voice_text=None):
    """
    Gathers every piece of state the UI needs into one dict, including
    the recommendation lookups — keeps main.py from having to import
    recommendations.py itself just to build this.
    """
    glasses_current_name = glasses_names[glasses_idx] if glasses_names and glasses_idx < len(glasses_names) else ""
    try:
        learned_count = feedback.count_ratings_for(face_shape, gender) if face_shape else 0
    except Exception as e:
        print(f"[ui] feedback count lookup failed: {e}")
        learned_count = 0
    return {
        'gender': gender, 'voice_status': voice_status, 'fps': fps,
        'face_shape': face_shape, 'face_shape_confidence': face_shape_confidence,
        'analyzing': analyzing, 'analysis_progress': analysis_progress,
        'near_edge': near_edge, 'cnn_busy': cnn_busy,
        'face_present': face_present, 'multiple_faces': multiple_faces,
        'rule_based_mode': rule_based_mode,
        'skin_tone': skin_tone, 'skin_swatch_rgb': skin_swatch_rgb,
        'skin_tone_low_confidence': skin_tone_low_confidence,
        'hair_recs': get_hair_rec(gender, face_shape),
        'grooming_tips': get_grooming_rec(gender, face_shape),
        'color_rec': get_color_rec(gender, skin_tone),
        'avoid_colors': get_avoid_colors(skin_tone),
        'glasses_types': get_glasses_rec(face_shape),
        'glasses_idx': glasses_idx, 'glasses_names': glasses_names,
        'glasses_current_name': glasses_current_name,
        'total_glasses': total_glasses,
        'is_recommended': glasses_idx in suggested,
        'learned_count': learned_count,
        'active_tab': active_tab,
        'mic_available': mic_available,
        'last_voice_text': last_voice_text,
    }


# ──────────────────────────────────────────
# Fallback: plain cv2 UI, used only if the TrueType font can't be
# loaded at all. Deliberately simple — this exists so a missing/
# corrupt font file can never crash the app, not to look good. Left
# unrestyled on purpose: it's the emergency safety net, not the
# showcase UI, and touching it just adds risk to a path that's meant
# to always work.
# ──────────────────────────────────────────
def _draw_ui_fallback(frame, state):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 45), (20, 20, 20), -1)
    cv2.putText(frame, f"AI Grooming Assistant | Gender: {state['gender'].title()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 1, cv2.LINE_AA)
    mic_on = state['voice_status'] in ('on', 'listening')
    cv2.putText(frame, "[MIC ON]" if mic_on else "[MIC OFF]", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if mic_on else (100, 100, 100), 1, cv2.LINE_AA)
    if state.get('rule_based_mode'):
        cv2.putText(frame, "[RULE-BASED MODE]", (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
    if state.get('multiple_faces'):
        cv2.putText(frame, "Multiple faces - using largest", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

    panel_x = w - 230
    cv2.rectangle(frame, (panel_x, 50), (w, h - 50), (20, 20, 20), -1)
    y = 75
    cv2.putText(frame, "FACE SHAPE", (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    y += 22
    if not state.get('face_present', True):
        cv2.putText(frame, "No face detected", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1, cv2.LINE_AA)
    elif state['analyzing']:
        collected, total = state['analysis_progress']
        cv2.putText(frame, f"Analyzing... {collected}/{total}", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
    else:
        fs = state['face_shape']
        cv2.putText(frame, fs or "Detecting...", (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 150) if fs else (100, 100, 100), 2, cv2.LINE_AA)
        if fs and state['face_shape_confidence'] is not None:
            y += 18
            cv2.putText(frame, f"{state['face_shape_confidence']:.0f}%", (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        if fs and state.get('learned_count', 0) > 0:
            y += 16
            cv2.putText(frame, f"Learned from {state['learned_count']} ratings", (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (74, 173, 176), 1, cv2.LINE_AA)

    y += 35
    cv2.putText(frame, "HAIRSTYLE", (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(frame, get_hair_rec_summary(state['gender'], state['face_shape']), (panel_x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    y += 35
    cv2.putText(frame, "SKIN TONE", (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(frame, state['skin_tone'] or "Detecting...", (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 200, 255) if state['skin_tone'] else (100, 100, 100), 2, cv2.LINE_AA)
    if state['skin_tone'] and state.get('skin_tone_low_confidence'):
        y += 16
        cv2.putText(frame, "Rough estimate - shadow detected", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, AMBER_BGR, 1, cv2.LINE_AA)
    if state['skin_tone']:
        y += 16
        cv2.putText(frame, "For best results, remove hats / even lighting", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (142, 148, 157), 1, cv2.LINE_AA)

    y += 35
    cv2.putText(frame, "OUTFIT COLORS", (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    y += 20
    for part in state['color_rec'].split(', '):
        cv2.putText(frame, part, (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        y += 16

    y += 15
    cv2.putText(frame, "GLASSES", (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    y += 20
    cv2.putText(frame, get_glasses_rec_summary(state['face_shape']), (panel_x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, h - 30), (w - 230, h), (20, 20, 20), -1)
    if state['glasses_names']:
        tag = " * REC" if state['is_recommended'] else ""
        cv2.putText(frame, f"Glasses {state['glasses_idx']+1}/{state['total_glasses']}: "
                            f"{state['glasses_current_name']}{tag}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def draw_notification(frame, text, h):
    cv2.putText(
        frame, text,
        (10, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 255), 2, cv2.LINE_AA
    )
    return frame


def draw_rating_prompt(frame, stage, suggestion_name):
    """
    Small modal-style overlay asking the user to rate the current top
    hairstyle/glasses suggestion 1-5 (Esc skips). Plain cv2, not PIL —
    this is a rare, transient, interactive overlay, not part of the
    cached chrome.
    """
    h, w = frame.shape[:2]
    box_w, box_h = 480, 90
    x1 = (w - box_w) // 2
    y1 = h - 170
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + box_w, y1 + box_h), (25, 28, 33), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), BLUE_ACCENT_BGR, 2, cv2.LINE_AA)

    label = "Hairstyle" if stage == 'hairstyle' else "Glasses"
    cv2.putText(frame, f"Rate this {label} suggestion: {suggestion_name}",
                (x1 + 16, y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Press 1-5 to rate  *  Esc to skip",
                (x1 + 16, y1 + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 184, 190), 1, cv2.LINE_AA)
    return frame


# ──────────────────────────────────────────
# Gender selection screen. Three tiers:
#   1. assets/ui/start_screen.png, if present -- the exact design image.
#   2. Otherwise, the drawn PIL screen (sparkle/gradient cards/icons).
#   3. Otherwise (no usable font either), the plain cv2 screen.
# All rendered ONCE (nothing on this screen animates on its own),
# then just redisplayed while waiting for a card click/key/hover.
# ──────────────────────────────────────────
START_SCREEN_PATH = str(ASSETS_DIR / 'ui' / 'start_screen.png')
START_MALE_FRAC   = (0.064, 0.377, 0.489, 0.723)
START_FEMALE_FRAC = (0.513, 0.377, 0.938, 0.723)

GENDER_CARD_W    = 576
GENDER_CARD_H    = 340
GENDER_CARD_Y    = 236
GENDER_CARD_GAP  = 48
GENDER_SIDE_MARGIN = (WINDOW_W - 2 * GENDER_CARD_W - GENDER_CARD_GAP) // 2
MALE_CARD_RECT   = (GENDER_SIDE_MARGIN, GENDER_CARD_Y,
                     GENDER_SIDE_MARGIN + GENDER_CARD_W, GENDER_CARD_Y + GENDER_CARD_H)
FEMALE_CARD_RECT = (MALE_CARD_RECT[2] + GENDER_CARD_GAP, GENDER_CARD_Y,
                     MALE_CARD_RECT[2] + GENDER_CARD_GAP + GENDER_CARD_W, GENDER_CARD_Y + GENDER_CARD_H)


def _draw_gender_card(bg_img, draw, fonts, rect, title, subtext, color_top, color_bottom, s):
    x1, y1, x2, y2 = (v * s for v in rect)
    w, h = x2 - x1, y2 - y1
    grad = _gradient_image(int(w), int(h), color_top, color_bottom)
    mask = Image.new('L', (int(w), int(h)), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=24 * s, fill=255)
    bg_img.paste(grad, (int(x1), int(y1)), mask)

    cx = x1 + w // 2
    if title == "MALE":
        _male_silhouette(draw, cx, y1 + 34 * s, 1.15 * s, TEXT_WHITE)
    else:
        _female_silhouette(draw, cx, y1 + 34 * s, 1.15 * s, TEXT_WHITE)

    title_font = fonts[('bold', 28 * s)]
    tw = draw.textlength(title, font=title_font)
    ty = y1 + 190 * s
    draw.text((cx - tw / 2, ty), title, font=title_font, fill=TEXT_WHITE)

    sub_font = fonts[('regular', 14 * s)]
    sw = draw.textlength(subtext, font=sub_font)
    draw.text((cx - sw / 2, ty + 40 * s), subtext, font=sub_font, fill=(232, 235, 245))

    _arrow_button(draw, x2 - 42 * s, y2 - 42 * s, 22 * s, (255, 255, 255), color_bottom)


def _render_gender_screen(fonts):
    """Entirely built at SS-times resolution, then downsampled once at the end (see _downsample)."""
    s = SS
    base = np.full((WINDOW_H * s, WINDOW_W * s, 3), BG_APP, dtype=np.float32)
    yy, xx = np.mgrid[0:WINDOW_H * s, 0:WINDOW_W * s].astype(np.float32)
    for (gx, gy), color, radius, strength in (
        ((0, 0), BLUE_1, 620 * s, 0.30),
        ((WINDOW_W * s, WINDOW_H * s), PURPLE_1, 620 * s, 0.28),
    ):
        dist = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2)
        glow = np.clip(1 - dist / radius, 0, 1) ** 2 * strength
        base += glow[:, :, None] * (np.array(color, dtype=np.float32) - base)
    bg = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), 'RGB')
    draw = ImageDraw.Draw(bg)
    cx = WINDOW_W * s // 2

    _sparkle(draw, cx, 46 * s, 12 * s, BLUE_1)

    title = "AI Grooming Assistant"
    title_font = fonts[('bold', 38 * s)]
    tw = draw.textlength(title, font=title_font)
    draw.text((cx - tw / 2, 66 * s), title, font=title_font, fill=TEXT_WHITE)
    line_y = (66 + 26) * s
    _gradient_line(draw, cx - tw / 2 - 120 * s, cx - tw / 2 - 20 * s, line_y, BG_APP, BLUE_1, thickness=2 * s)
    _gradient_line(draw, cx + tw / 2 + 20 * s, cx + tw / 2 + 120 * s, line_y, PURPLE_1, BG_APP, thickness=2 * s)

    subtitle = "Discover Your Best Look - Powered by AI"
    sf = fonts[('regular', 18 * s)]
    sw = draw.textlength(subtitle, font=sf)
    draw.text((cx - sw / 2, 120 * s), subtitle, font=sf, fill=TEXT_DIM)

    desc = "Get personalized style recommendations based on your face shape and skin tone"
    df = fonts[('regular', 13 * s)]
    dw = draw.textlength(desc, font=df)
    draw.text((cx - dw / 2, 150 * s), desc, font=df, fill=TEXT_DIM)

    _draw_gender_card(bg, draw, fonts, MALE_CARD_RECT, "MALE",
                       "Find your perfect style and look", BLUE_1, BLUE_2, s)
    _draw_gender_card(bg, draw, fonts, FEMALE_CARD_RECT, "FEMALE",
                       "Enhance your style with AI", PURPLE_1, PURPLE_2, s)

    features = [
        ("Face Shape Analysis", _face_outline_icon, BLUE_1),
        ("Skin Tone Detection", _droplet_icon, PURPLE_1),
        ("Smart Recommendations", _sparkle, BLUE_1),
        ("AR Visualization & Voice Assistant", _glasses_icon, PURPLE_1),
    ]
    fy = (GENDER_CARD_Y + GENDER_CARD_H + 46) * s
    cell_w = WINDOW_W * s // 4
    ffont = fonts[('regular', 12 * s)]
    for i, (label, icon_fn, color) in enumerate(features):
        fx = cell_w * i + cell_w // 2
        draw.ellipse([fx - 26 * s, fy - 26 * s, fx + 26 * s, fy + 26 * s], outline=color, width=2 * s)
        icon_fn(draw, fx, fy, 14 * s, color)
        lines = _wrap_text(draw, label, ffont, cell_w - 40 * s)
        ly = fy + 38 * s
        for line in lines:
            lw = draw.textlength(line, font=ffont)
            draw.text((fx - lw / 2, ly), line, font=ffont, fill=TEXT_DIM)
            ly += 16 * s

    _draw_spaced_text(draw, cx, (WINDOW_H - 20) * s, "LOOK GOOD  •  FEEL CONFIDENT",
                       fonts[('regular', 11 * s)], TEXT_DIM, spacing=3 * s)

    bg = _downsample(bg)
    return cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)


def _show_gender_selection_fallback():
    """Plain cv2 version, used only if the TrueType fonts couldn't be loaded at all."""
    selected = [None]
    W, H = 640, 400

    while selected[0] is None:
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        for i in range(H):
            ratio = i / H
            frame[i, :] = [int(20 + ratio * 10), int(20 + ratio * 10), int(40 + ratio * 20)]

        cv2.putText(frame, "AI Grooming Assistant", (80, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 150), 2, cv2.LINE_AA)
        cv2.putText(frame, "Select Your Gender to Continue", (100, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.rectangle(frame, (80, 180), (260, 270), (200, 150, 50), -1)
        cv2.rectangle(frame, (80, 180), (260, 270), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "MALE", (130, 235), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.rectangle(frame, (360, 180), (560, 270), (180, 50, 180), -1)
        cv2.rectangle(frame, (360, 180), (560, 270), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "FEMALE", (390, 235), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, "Click on your gender", (200, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow("AI Grooming Assistant", frame)

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if 80 <= x <= 260 and 180 <= y <= 270:
                    selected[0] = 'male'
                elif 360 <= x <= 560 and 180 <= y <= 270:
                    selected[0] = 'female'

        cv2.setMouseCallback("AI Grooming Assistant", on_click)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord('m'), ord('M')):
            selected[0] = 'male'
        elif key in (ord('f'), ord('F')):
            selected[0] = 'female'
        elif key == ord('q'):
            selected[0] = 'male'
            break

    cv2.destroyAllWindows()
    print(f"Gender selected: {selected[0]}")
    return selected[0]


def _show_gender_selection_drawn():
    """Middle tier: the drawn PIL screen, used when the design PNG is missing but fonts loaded fine."""
    fonts, ok = _load_fonts()
    if not ok:
        return _show_gender_selection_fallback()

    frame = _render_gender_screen(fonts)
    selected = [None]
    win = "AI Grooming Assistant"
    cv2.imshow(win, frame)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if _point_in_rect(x, y, MALE_CARD_RECT):
                selected[0] = 'male'
            elif _point_in_rect(x, y, FEMALE_CARD_RECT):
                selected[0] = 'female'

    cv2.setMouseCallback(win, on_click)

    while selected[0] is None:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('m'), ord('M')):
            selected[0] = 'male'
        elif key in (ord('f'), ord('F')):
            selected[0] = 'female'
        elif key == ord('q'):
            selected[0] = 'male'
            break

    cv2.destroyAllWindows()
    print(f"Gender selected: {selected[0]}")
    return selected[0]


def _fit_image_letterboxed(img_bgr, target_w, target_h, bg_color_bgr):
    h, w = img_bgr.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=interp)
    canvas = np.full((target_h, target_w, 3), bg_color_bgr, dtype=np.uint8)
    x_off, y_off = (target_w - nw) // 2, (target_h - nh) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    return canvas, (x_off, y_off, nw, nh)


def _start_card_rect_px(placement, frac):
    x_off, y_off, nw, nh = placement
    fx1, fy1, fx2, fy2 = frac
    return (x_off + fx1 * nw, y_off + fy1 * nh, x_off + fx2 * nw, y_off + fy2 * nh)


def _hit_test_start_card(x, y, placement):
    if _point_in_rect(x, y, _start_card_rect_px(placement, START_MALE_FRAC)):
        return 'male'
    if _point_in_rect(x, y, _start_card_rect_px(placement, START_FEMALE_FRAC)):
        return 'female'
    return None


def _make_hover_glow(rect, color_rgb, margin=34, blur=10):
    """Real Gaussian-blurred glow ring around `rect` (window px) -- built once per hover-in, then cached by the caller."""
    x1, y1, x2, y2 = rect
    w, h = int(x2 - x1) + margin * 2, int(y2 - y1) + margin * 2
    layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle([margin, margin, margin + (x2 - x1), margin + (y2 - y1)], radius=26,
                         outline=color_rgb + (255,), width=6)
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    return layer, (int(x1) - margin, int(y1) - margin)


def _show_gender_selection_from_image(img_bgr):
    """Top tier: the exact design PNG at assets/ui/start_screen.png, with hover glow + click zones from the given fractions."""
    base_frame, placement = _fit_image_letterboxed(img_bgr, WINDOW_W, WINDOW_H, START_LETTERBOX_BGR)
    base_rgba = Image.fromarray(cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB)).convert('RGBA')

    win = "AI Grooming Assistant"
    selected = [None]
    hovered = [None]
    glow_cache = {}

    def composite(hover_key):
        if hover_key is None:
            return base_frame
        if hover_key not in glow_cache:
            rect = _start_card_rect_px(placement, START_MALE_FRAC if hover_key == 'male' else START_FEMALE_FRAC)
            color = BLUE_1 if hover_key == 'male' else PURPLE_1
            glow_cache[hover_key] = _make_hover_glow(rect, color)
        layer, (px, py) = glow_cache[hover_key]
        composed = base_rgba.copy()
        composed.alpha_composite(layer, (px, py))
        return cv2.cvtColor(np.array(composed.convert('RGB')), cv2.COLOR_RGB2BGR)

    cv2.imshow(win, base_frame)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            card = _hit_test_start_card(x, y, placement)
            if card != hovered[0]:
                hovered[0] = card
                cv2.imshow(win, composite(card))
        elif event == cv2.EVENT_LBUTTONDOWN:
            card = _hit_test_start_card(x, y, placement)
            if card:
                selected[0] = card

    cv2.setMouseCallback(win, on_mouse)

    while selected[0] is None:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('m'), ord('M')):
            selected[0] = 'male'
        elif key in (ord('f'), ord('F')):
            selected[0] = 'female'
        elif key == ord('q'):
            selected[0] = 'male'
            break

    cv2.destroyAllWindows()
    print(f"Gender selected: {selected[0]}")
    return selected[0]


def show_gender_selection():
    """
    Show the gender selection screen before the app starts. Uses
    assets/ui/start_screen.png if present, falls back to the drawn PIL
    screen, falls back again to a plain cv2 screen if no font loaded
    at all -- see the module comment above. Returns 'male' or 'female'.
    """
    if os.path.exists(START_SCREEN_PATH):
        img = cv2.imread(START_SCREEN_PATH, cv2.IMREAD_COLOR)
        if img is not None:
            return _show_gender_selection_from_image(img)
        print(f"[ui] Couldn't read {START_SCREEN_PATH}; falling back to the drawn start screen.")
    return _show_gender_selection_drawn()

"""
On-screen UI: a 1280x720 layout with the camera feed on the left and a
tab-driven card panel on the right, rendered with PIL (TrueType text,
rounded cards, gradients) and composited onto the OpenCV frame.

Visual language: near-black navy background, blue accent for
face-shape/male-related elements, purple/pink accent for
skin-tone/female-related elements (see the color constants below).

Performance: the top bar, right panel, and bottom bar are each cached
as plain BGR images and only re-rendered when the data they show
actually changes; every frame just pastes the cached images in with
numpy slicing. The video feed's glowing border is drawn every frame
with plain cv2.rectangle calls (cheap — only touches the pixels on the
line itself); a true rounded-corner clip was tried first but measured
~3.7ms/frame even precomputed as a boolean mask, so the video's
corners stay square instead (see the comment in UIRenderer.render).
Only the video area itself (camera feed + AR glasses, already drawn by
the caller) and a few small live overlays (badges, hints, the CNN-busy
pulse dot) are touched every frame, using cheap plain cv2 calls rather
than PIL.

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
from PIL import Image, ImageDraw, ImageFont

from recommendations import (
    get_hair_rec, get_hair_rec_summary, get_grooming_rec, get_glasses_rec,
    get_glasses_rec_summary, get_color_rec, get_avoid_colors, get_color_swatch,
)
import feedback

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

# Border thickness for the live camera feed's glowing outline.
VIDEO_BORDER_THICK  = 3
VIDEO_HALO_THICK    = 2

# ──────────────────────────────────────────
# Colors (RGB, since PIL draws in RGB — converted to BGR once per
# cached image / constant when handed back to OpenCV)
# ──────────────────────────────────────────
BG_APP     = (10, 14, 26)      # #0A0E1A — app background
BG_CARD    = (18, 23, 42)      # #12172A — card background
BLUE_1     = (46, 158, 255)    # #2E9EFF
BLUE_2     = (27, 111, 224)    # #1B6FE0
PURPLE_1   = (168, 85, 247)    # #A855F7
PURPLE_2   = (224, 86, 253)    # #E056FD
TEXT_WHITE = (255, 255, 255)
TEXT_DIM   = (154, 163, 184)   # #9AA3B8
GREEN      = (34, 197, 94)     # #22C55E
AMBER      = (230, 172, 60)    # not in the reference palette, kept for low-confidence/warning text
RED        = (222, 90, 90)     # kept, for a low confidence bar
PILL_BG    = (30, 35, 54)

# Generic accent aliases: blue = face-shape/male-related, purple = skin-tone/female-related
BLUE_ACCENT   = BLUE_1
PURPLE_ACCENT = PURPLE_1


def _mix(c1, c2, t):
    return tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))


BLUE_HALO_RGB = _mix(BG_APP, BLUE_1, 0.45)  # faint version of the border color, for the halo ring

BLUE_ACCENT_BGR   = BLUE_ACCENT[::-1]
PURPLE_ACCENT_BGR = PURPLE_ACCENT[::-1]
BLUE_HALO_BGR      = BLUE_HALO_RGB[::-1]
AMBER_BGR    = AMBER[::-1]
GREEN_BGR    = GREEN[::-1]
BG_APP_BGR   = BG_APP[::-1]


def _confidence_color(pct):
    if pct >= 60:
        return GREEN
    elif pct >= 40:
        return AMBER
    return RED


# ──────────────────────────────────────────
# Font loading (bundled TrueType, downloaded on first run if missing)
# ──────────────────────────────────────────
FONT_PATH = os.path.join('assets', 'fonts', 'Inter-Variable.ttf')
FONT_URL  = (
    'https://raw.githubusercontent.com/google/fonts/main/'
    'ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf'
)


def _download_font():
    try:
        os.makedirs(os.path.dirname(FONT_PATH), exist_ok=True)
        print("[ui] Downloading UI font...")
        urllib.request.urlretrieve(FONT_URL, FONT_PATH)
        print("[ui] Font ready!")
    except Exception as e:
        print(f"[ui] Couldn't download font ({e}); will use the simpler cv2 UI.")


class _FontDict(dict):
    """
    Dict of ('regular'|'bold', size) -> ImageFont. If some code asks for
    a size that was never preloaded (e.g. a new UI element added later
    using a size _load_fonts() doesn't know about), falling through to
    a KeyError would crash render() on *every* frame -- and since
    render() runs inside main.py's per-frame try/except, that silently
    skips cv2.imshow()/waitKey() for the rest of the session too,
    freezing the window instead of just looking wrong. Falling back to
    the closest already-loaded size of the same style keeps the app
    running (a slightly-off font size is a cosmetic issue, not a crash)
    while still printing a one-time warning so the real fix -- adding
    the size to _load_fonts()'s preloaded sizes -- doesn't get missed.
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
                  f"Add {size} to _load_fonts()'s preloaded sizes to fix this properly.")
            self._warned.add(key)
        return self[(style, closest)]


def _load_fonts():
    """Returns (fonts_dict, ok). fonts_dict maps ('regular'|'bold', size) -> ImageFont."""
    if not os.path.exists(FONT_PATH):
        _download_font()
    if not os.path.exists(FONT_PATH):
        return {}, False
    try:
        fonts = _FontDict()
        # Every (style, size) actually used anywhere below (both the
        # camera-screen cards and the gender-selection screen) must be
        # listed here.
        for size in (11, 12, 13, 14, 18, 26, 28, 38):
            regular = ImageFont.truetype(FONT_PATH, size)
            bold    = ImageFont.truetype(FONT_PATH, size)
            try:
                bold.set_variation_by_name('Bold')
            except Exception:
                pass  # font has no named instances (static file) — bold just matches regular
            fonts[('regular', size)] = regular
            fonts[('bold', size)]    = bold
        return fonts, True
    except Exception as e:
        print(f"[ui] Couldn't load UI font ({e}); will use the simpler cv2 UI.")
        return {}, False


# ──────────────────────────────────────────
# Low-level PIL drawing helpers
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


def _card(draw, x, y, w, h, radius=14, fill=BG_CARD, border=None):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=fill)
    if border:
        draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, outline=border, width=1)


def _pill(draw, x, y, text, font, fg, bg, pad_x=10, pad_y=5):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    w, h = tw + pad_x * 2, th + pad_y * 2
    draw.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=bg)
    draw.text((x + pad_x - bbox[0], y + pad_y - bbox[1]), text, font=font, fill=fg)
    return w, h


def _progress_bar(draw, x, y, w, h, frac, color, bg=(40, 45, 63)):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=bg)
    fw = int(w * max(0.0, min(frac, 1.0)))
    if fw > h:
        draw.rounded_rectangle([x, y, x + fw, y + h], radius=h // 2, fill=color)


def _circle(draw, cx, cy, r, color, outline=None):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=outline,
                 width=1 if outline else 0)


def _check_badge(draw, cx, cy, r=9, color=GREEN):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    draw.line([(cx - r * 0.5, cy), (cx - r * 0.1, cy + r * 0.4)], fill=(255, 255, 255), width=2)
    draw.line([(cx - r * 0.1, cy + r * 0.4), (cx + r * 0.55, cy - r * 0.35)], fill=(255, 255, 255), width=2)


def _gradient_image(w, h, color_top, color_bottom):
    """One-time vertical gradient, used only for gender-screen cards/background (never per-frame)."""
    top = np.array(color_top, dtype=np.float32)
    bottom = np.array(color_bottom, dtype=np.float32)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    col = (top[None, :] * (1 - t) + bottom[None, :] * t).astype(np.uint8)  # (h, 3)
    arr = np.repeat(col[:, None, :], w, axis=1)
    return Image.fromarray(arr, 'RGB')


def _rounded_gradient_card(w, h, color_top, color_bottom, radius=20):
    grad = _gradient_image(w, h, color_top, color_bottom)
    mask = Image.new('L', (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    return grad, mask


def _glow_background(w, h):
    """
    Soft radial glow in two corners, blended onto the near-black app
    background -- one-time only (the gender screen is rendered once,
    not per-frame). The main camera screen doesn't use this: the
    video/panel/bars cover almost the entire canvas there, so a corner
    glow would barely be visible and isn't worth computing every frame.
    """
    base = np.full((h, w, 3), BG_APP, dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for (gx, gy), color, radius, strength in (
        ((0, 0), BLUE_1, 620, 0.30),
        ((w, h), PURPLE_1, 620, 0.28),
    ):
        dist = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2)
        glow = np.clip(1 - dist / radius, 0, 1) ** 2 * strength
        base += glow[:, :, None] * (np.array(color, dtype=np.float32) - base)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), 'RGB')


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
    draw.ellipse([cx - r, cy - r * 1.15, cx + r, cy + r * 1.15], outline=color, width=2)
    eye_r = max(1, r * 0.08)
    draw.ellipse([cx - r * 0.4 - eye_r, cy - eye_r, cx - r * 0.4 + eye_r, cy + eye_r], fill=color)
    draw.ellipse([cx + r * 0.4 - eye_r, cy - eye_r, cx + r * 0.4 + eye_r, cy + eye_r], fill=color)
    draw.arc([cx - r * 0.4, cy + r * 0.15, cx + r * 0.4, cy + r * 0.55], start=20, end=160, fill=color, width=2)


def _droplet_icon(draw, cx, cy, r, color):
    draw.ellipse([cx - r, cy - r * 0.2, cx + r, cy + r * 1.2], fill=color)
    draw.polygon([(cx, cy - r * 1.3), (cx - r * 0.6, cy), (cx + r * 0.6, cy)], fill=color)


def _glasses_icon(draw, cx, cy, r, color):
    lens_r = r * 0.42
    gap = r * 0.3
    draw.ellipse([cx - gap - 2 * lens_r, cy - lens_r, cx - gap, cy + lens_r], outline=color, width=2)
    draw.ellipse([cx + gap, cy - lens_r, cx + gap + 2 * lens_r, cy + lens_r], outline=color, width=2)
    draw.line([(cx - gap, cy), (cx + gap, cy)], fill=color, width=2)


def _mic_icon(draw, cx, cy, size, color):
    w = size * 0.5
    draw.rounded_rectangle([cx - w / 2, cy - size, cx + w / 2, cy], radius=w / 2, fill=color)
    draw.arc([cx - size * 0.8, cy - size * 0.5, cx + size * 0.8, cy + size * 0.5],
              start=20, end=160, fill=color, width=2)
    draw.line([(cx, cy + size * 0.5), (cx, cy + size * 0.8)], fill=color, width=2)
    draw.line([(cx - size * 0.35, cy + size * 0.8), (cx + size * 0.35, cy + size * 0.8)], fill=color, width=2)


def _gear_icon(draw, cx, cy, r, color):
    draw.ellipse([cx - r * 0.55, cy - r * 0.55, cx + r * 0.55, cy + r * 0.55], outline=color, width=2)
    for i in range(8):
        ang = i * math.pi / 4
        x1, y1 = cx + r * 0.6 * math.cos(ang), cy + r * 0.6 * math.sin(ang)
        x2, y2 = cx + r * math.cos(ang), cy + r * math.sin(ang)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=3)


def _point_in_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


# ──────────────────────────────────────────
# Right panel cards
# ──────────────────────────────────────────
def _draw_face_shape_card(draw, x, y, w, fonts, state):
    pad = 12
    show_learned = (not state['analyzing']) and state['face_shape'] and state['learned_count'] > 0
    if not state['face_present']:
        height = 32 + 22 + pad
    elif state['analyzing']:
        height = 32 + 22 + 8 + 10 + (18 if state['near_edge'] else 0) + pad
    else:
        has_conf = state['face_shape'] and state['face_shape_confidence'] is not None
        height = 32 + 34 + (8 + 6 + 18 if has_conf else 0) + (18 if show_learned else 0) + pad

    _card(draw, x, y, w, height, border=BLUE_ACCENT)
    draw.text((x + pad, y + 10), "FACE SHAPE", font=fonts[('bold', 12)], fill=TEXT_DIM)
    _face_outline_icon(draw, x + w - pad - 10, y + 16, 9, BLUE_ACCENT)
    cy = y + 32

    if not state['face_present']:
        draw.text((x + pad, cy), "No face detected", font=fonts[('regular', 18)], fill=TEXT_DIM)
    elif state['analyzing']:
        draw.text((x + pad, cy), "Analyzing...", font=fonts[('regular', 18)], fill=BLUE_ACCENT)
        cy += 26
        collected, total = state['analysis_progress']
        _progress_bar(draw, x + pad, cy, w - 2 * pad, 8, collected / max(total, 1), BLUE_ACCENT)
        cy += 8 + 10
        if state['near_edge']:
            draw.text((x + pad, cy), "Move to center", font=fonts[('regular', 12)], fill=AMBER)
    else:
        label = state['face_shape'] or "Detecting..."
        draw.text((x + pad, cy), label, font=fonts[('bold', 26)],
                  fill=TEXT_WHITE if state['face_shape'] else TEXT_DIM)
        if state['face_shape'] and state['face_shape_confidence'] is not None:
            label_w = draw.textlength(label, font=fonts[('bold', 26)])
            _check_badge(draw, x + pad + label_w + 18, cy + 13)
        cy += 34
        conf = state['face_shape_confidence']
        if state['face_shape'] and conf is not None:
            _progress_bar(draw, x + pad, cy, w - 2 * pad, 8, conf / 100, _confidence_color(conf))
            cy += 8 + 6
            draw.text((x + pad, cy), f"{conf:.0f}% confidence (locked)",
                      font=fonts[('regular', 12)], fill=TEXT_DIM)
            cy += 18
        if show_learned:
            draw.text((x + pad, cy), f"Learned from {state['learned_count']} ratings",
                      font=fonts[('regular', 12)], fill=BLUE_ACCENT)

    return y + height + CARD_GAP


def _draw_skin_tone_card(draw, x, y, w, fonts, state):
    pad = 12
    show_hint = bool(state['skin_tone']) or state.get('skin_tone_low_confidence')
    height = 32 + 40 + pad + (18 if show_hint else 0)
    _card(draw, x, y, w, height, border=PURPLE_ACCENT)
    draw.text((x + pad, y + 10), "SKIN TONE", font=fonts[('bold', 12)], fill=TEXT_DIM)
    cy = y + 32 + 6
    if state['skin_tone']:
        swatch = state['skin_swatch_rgb'] or (150, 150, 150)
        _circle(draw, x + pad + 14, cy + 14, 14, swatch, outline=PURPLE_ACCENT)
        draw.text((x + pad + 38, cy + 4), state['skin_tone'], font=fonts[('regular', 18)], fill=TEXT_WHITE)
        if state.get('skin_tone_low_confidence'):
            draw.text((x + pad + 38, cy + 26), "Rough estimate — shadow detected", font=fonts[('regular', 12)], fill=AMBER)
    else:
        draw.text((x + pad, cy + 4), "Detecting...", font=fonts[('regular', 18)], fill=TEXT_DIM)
    if show_hint:
        draw.text((x + pad, y + height - 18), "For best results, remove hats and ensure even lighting",
                  font=fonts[('regular', 11)], fill=TEXT_DIM)
    return y + height + CARD_GAP


def _draw_hairstyle_card(draw, x, y, w, fonts, state):
    pad = 12
    content_w = w - 2 * pad
    recs = state['hair_recs'][:3]
    items = []
    height = 32
    for rec in recs:
        why_lines = _wrap_text(draw, rec['why'], fonts[('regular', 12)], content_w - 4)
        item_h = 20 + len(why_lines) * 15 + 6
        items.append((rec, why_lines))
        height += item_h
    if not recs:
        height += 20
    height += pad

    _card(draw, x, y, w, height, border=BLUE_ACCENT)
    draw.text((x + pad, y + 10), "HAIRSTYLE", font=fonts[('bold', 12)], fill=TEXT_DIM)
    cy = y + 32
    if not recs:
        draw.text((x + pad, cy), "---", font=fonts[('regular', 14)], fill=TEXT_DIM)
    for rec, why_lines in items:
        name_color = TEXT_DIM if rec.get('less_popular') else TEXT_WHITE
        draw.text((x + pad, cy), rec['name'], font=fonts[('regular', 14)], fill=name_color)
        if rec.get('less_popular'):
            name_w = draw.textlength(rec['name'], font=fonts[('regular', 14)])
            draw.text((x + pad + name_w + 8, cy + 2), "less popular", font=fonts[('regular', 12)], fill=AMBER)
        cy += 20
        for line in why_lines:
            draw.text((x + pad + 4, cy), line, font=fonts[('regular', 12)], fill=TEXT_DIM)
            cy += 15
        cy += 6
    return y + height + CARD_GAP


def _draw_grooming_card(draw, x, y, w, fonts, state):
    pad = 12
    content_w = w - 2 * pad - 12
    tips = state['grooming_tips']
    tip_lines = []
    height = 30
    for tip in tips:
        wrapped = _wrap_text(draw, tip, fonts[('regular', 14)], content_w)
        tip_lines.append(wrapped)
        height += len(wrapped) * 19 + 5
    if not tips:
        height += 18
    height += pad

    _card(draw, x, y, w, height, border=BLUE_ACCENT)
    draw.text((x + pad, y + 10), "GROOMING", font=fonts[('bold', 12)], fill=TEXT_DIM)
    cy = y + 30
    if not tips:
        draw.text((x + pad, cy), "---", font=fonts[('regular', 14)], fill=TEXT_DIM)
    for wrapped in tip_lines:
        for i, line in enumerate(wrapped):
            prefix = "• " if i == 0 else "   "
            draw.text((x + pad, cy), prefix + line, font=fonts[('regular', 14)], fill=TEXT_WHITE)
            cy += 19
        cy += 5
    return y + height + CARD_GAP


def _wrap_color_names(draw, names, font, max_w, dim_label=None):
    """Groups color names into rows that fit within max_w — wraps rather than dropping overflow items."""
    lines, current, cx = [], [], (draw.textlength(dim_label, font=font) + 6 if dim_label else 0)
    r = 6 if dim_label else 7
    for name in names:
        item_w = 2 * r + 6 + draw.textlength(name, font=font) + 12
        if cx + item_w > max_w and current:
            lines.append(current)
            current, cx = [], 0
        current.append(name)
        cx += item_w
    if current:
        lines.append(current)
    return lines or [[]]


def _draw_color_lines(draw, x, y, lines, font, name_color, dim_label=None, row_h=20):
    r = 6 if dim_label else 7
    for i, line_names in enumerate(lines):
        cx = x
        if i == 0 and dim_label:
            draw.text((cx, y), dim_label, font=font, fill=TEXT_DIM)
            cx += draw.textlength(dim_label, font=font) + 6
        for name in line_names:
            rgb = get_color_swatch(name)
            _circle(draw, cx + r, y + r + 1, r, rgb, outline=(70, 74, 80))
            cx += 2 * r + 6
            draw.text((cx, y), name, font=font, fill=name_color)
            cx += draw.textlength(name, font=font) + 12
        y += row_h
    return y


def _draw_outfit_card(draw, x, y, w, fonts, state):
    pad = 12
    content_w = w - 2 * pad
    font = fonts[('regular', 12)]
    rec_names   = [c.strip() for c in state['color_rec'].split(',')] if state['color_rec'] != '---' else []
    avoid_names = [c.strip() for c in state['avoid_colors'].split(',')] if state['avoid_colors'] != '---' else []

    wear_lines  = _wrap_color_names(draw, rec_names, font, content_w)
    avoid_lines = _wrap_color_names(draw, avoid_names, font, content_w, dim_label="Avoid:") if avoid_names else []

    height = 30 + len(wear_lines) * 20 + len(avoid_lines) * 20 + pad
    _card(draw, x, y, w, height, border=BLUE_ACCENT)
    draw.text((x + pad, y + 10), "OUTFIT COLORS", font=fonts[('bold', 12)], fill=TEXT_DIM)
    cy = y + 30
    cy = _draw_color_lines(draw, x + pad, cy, wear_lines, font, TEXT_WHITE)
    if avoid_lines:
        _draw_color_lines(draw, x + pad, cy, avoid_lines, font, TEXT_DIM, dim_label="Avoid:")
    return y + height + CARD_GAP


def _draw_glasses_card(draw, x, y, w, fonts, state):
    pad = 12
    content_w = w - 2 * pad
    recs = state['glasses_types'][:3]
    items = []
    height = 30
    for rec in recs:
        why_lines = _wrap_text(draw, rec['why'], fonts[('regular', 12)], content_w - 4)
        items.append((rec, why_lines))
        height += 18 + len(why_lines) * 15 + 5
    if not recs:
        height += 18
    info_line = bool(state['glasses_names'])
    height += (18 if info_line else 0) + pad

    _card(draw, x, y, w, height, border=BLUE_ACCENT)
    draw.text((x + pad, y + 10), "GLASSES", font=fonts[('bold', 12)], fill=TEXT_DIM)

    if state['is_recommended']:
        badge_text = "REC"
        bw = draw.textlength(badge_text, font=fonts[('bold', 12)]) + 14
        draw.rounded_rectangle([x + w - pad - bw, y + 8, x + w - pad, y + 28], radius=10, fill=GREEN)
        draw.text((x + w - pad - bw + 7, y + 11), badge_text, font=fonts[('bold', 12)], fill=(15, 25, 18))

    cy = y + 30
    if not recs:
        draw.text((x + pad, cy), "---", font=fonts[('regular', 14)], fill=TEXT_DIM)
    for rec, why_lines in items:
        name_color = TEXT_DIM if rec.get('less_popular') else TEXT_WHITE
        draw.text((x + pad, cy), rec['name'], font=fonts[('regular', 14)], fill=name_color)
        if rec.get('less_popular'):
            name_w = draw.textlength(rec['name'], font=fonts[('regular', 14)])
            draw.text((x + pad + name_w + 8, cy + 2), "less popular", font=fonts[('regular', 12)], fill=AMBER)
        cy += 18
        for line in why_lines:
            draw.text((x + pad + 4, cy), line, font=fonts[('regular', 12)], fill=TEXT_DIM)
            cy += 15
        cy += 5
    if info_line:
        info = f"{state['glasses_idx'] + 1}/{state['total_glasses']}  {state['glasses_current_name']}"
        draw.text((x + pad, cy), info, font=fonts[('regular', 12)], fill=TEXT_DIM)
    return y + height + CARD_GAP


def _draw_voice_card(draw, x, y, w, fonts, state):
    pad = 12
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

    height = 32 + 26 + (18 if last_text else 0) + 15 + 17 + len(commands) * 17 + pad
    _card(draw, x, y, w, height, border=BLUE_ACCENT)
    draw.text((x + pad, y + 10), "VOICE ASSISTANT", font=fonts[('bold', 12)], fill=TEXT_DIM)
    _mic_icon(draw, x + w - pad - 6, y + 18, 8, BLUE_ACCENT)
    cy = y + 32
    draw.text((x + pad, cy), status_text, font=fonts[('regular', 18)], fill=status_color)
    cy += 26
    if last_text:
        draw.text((x + pad, cy), f'Heard: "{last_text}"', font=fonts[('regular', 12)], fill=TEXT_DIM)
        cy += 18
    cy += 15
    draw.text((x + pad, cy), "Try saying:", font=fonts[('regular', 12)], fill=TEXT_DIM)
    cy += 17
    for c in commands:
        draw.text((x + pad + 4, cy), c, font=fonts[('regular', 12)], fill=TEXT_WHITE)
        cy += 17
    return y + height + CARD_GAP


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
    x, w = CARD_MARGIN, PANEL_W - 2 * CARD_MARGIN
    # Two passes aren't needed: each card function measures its own
    # content height before drawing, so we just need a canvas tall
    # enough for the worst case (comfortably above it — this is just
    # an initial buffer size, not the final image; it gets cropped to
    # the real content height below, and the caller scales the result
    # down to fit the visible panel area if it's still too tall).
    canvas_h = 1400
    img = Image.new('RGB', (PANEL_W, canvas_h), BG_APP)
    draw = ImageDraw.Draw(img)
    y = CARD_MARGIN
    tab = state.get('active_tab', 'face')
    if tab == 'face':
        y = _draw_face_shape_card(draw, x, y, w, fonts, state)
        y = _draw_skin_tone_card(draw, x, y, w, fonts, state)
    elif tab == 'hairstyle':
        y = _draw_hairstyle_card(draw, x, y, w, fonts, state)
        y = _draw_grooming_card(draw, x, y, w, fonts, state)
    elif tab == 'glasses':
        y = _draw_glasses_card(draw, x, y, w, fonts, state)
    elif tab == 'outfit':
        y = _draw_outfit_card(draw, x, y, w, fonts, state)
    elif tab == 'voice':
        y = _draw_voice_card(draw, x, y, w, fonts, state)
    img = img.crop((0, 0, PANEL_W, min(y, canvas_h)))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _render_top_bar(fonts, state):
    img = Image.new('RGB', (WINDOW_W, TOP_BAR_H), BG_APP)
    draw = ImageDraw.Draw(img)

    # App icon + title
    _sparkle(draw, 26, TOP_BAR_H // 2, 9, BLUE_1)
    title = "AI Grooming Assistant"
    title_font = fonts[('bold', 18)]
    draw.text((44, TOP_BAR_H // 2 - 12), title, font=title_font, fill=TEXT_WHITE)
    title_end_x = 44 + draw.textlength(title, font=title_font)

    if state.get('rule_based_mode'):
        # Kept visible on screen (not just printed to the console) --
        # stress test case 7 needs a visible badge when the trained
        # model files are missing and the app falls back to the
        # geometric rule-based classifier.
        _pill(draw, title_end_x + 16, TOP_BAR_H // 2 - 11, "Rule-based mode",
              fonts[('regular', 11)], AMBER, PILL_BG)

    # Gender pill-style toggle, centered. Display-only: gender is fixed
    # for the session once chosen on the selection screen, so this
    # isn't a live switch -- it just shows which side is active,
    # colored to match the male/female accent.
    is_male = state['gender'] == 'male'
    seg_w, seg_h = 56, 26
    px = WINDOW_W // 2 - seg_w
    py = (TOP_BAR_H - seg_h) // 2
    draw.rounded_rectangle([px, py, px + 2 * seg_w, py + seg_h], radius=seg_h // 2, fill=PILL_BG)
    active_x = px if is_male else px + seg_w
    draw.rounded_rectangle([active_x, py, active_x + seg_w, py + seg_h], radius=seg_h // 2,
                            fill=BLUE_ACCENT if is_male else PURPLE_ACCENT)
    mf = fonts[('regular', 11)]
    draw.text((px + seg_w / 2 - draw.textlength("MALE", font=mf) / 2, py + 7), "MALE",
              font=mf, fill=TEXT_WHITE if is_male else TEXT_DIM)
    draw.text((px + seg_w * 1.5 - draw.textlength("FEMALE", font=mf) / 2, py + 7), "FEMALE",
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

    vf = fonts[('regular', 12)]
    vw = draw.textlength(vtext, font=vf)
    gear_cx = WINDOW_W - 24
    vx = gear_cx - 22 - vw
    _mic_icon(draw, vx - 14, TOP_BAR_H // 2, 8, vcolor)
    draw.text((vx, TOP_BAR_H // 2 - 7), vtext, font=vf, fill=vcolor)
    _gear_icon(draw, gear_cx, TOP_BAR_H // 2, 11, TEXT_DIM)  # decorative, no settings menu exists

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _render_bottom_bar(fonts, state):
    img = Image.new('RGB', (WINDOW_W, BOTTOM_BAR_H), BG_APP)
    draw = ImageDraw.Draw(img)

    active = state.get('active_tab', 'face')
    seg_w = WINDOW_W / len(TAB_KEYS)
    tf_active = fonts[('bold', 13)]
    tf        = fonts[('regular', 13)]
    for i, key in enumerate(TAB_KEYS):
        cx = int(seg_w * i + seg_w / 2)
        label = TAB_LABELS[key]
        is_active = key == active
        color = BLUE_ACCENT if is_active else TEXT_DIM
        font = tf_active if is_active else tf
        lw = draw.textlength(label, font=font)
        draw.text((cx - lw / 2, 12), label, font=font, fill=color)
        if is_active:
            draw.rounded_rectangle([cx - lw / 2 - 4, 32, cx + lw / 2 + 4, 34], radius=1, fill=BLUE_ACCENT)

    draw.line([(0, BOTTOM_TAB_H), (WINDOW_W, BOTTOM_TAB_H)], fill=BG_CARD, width=1)

    tag_font = fonts[('regular', 11)]
    tag = "Better Style - Better You"
    tw = draw.textlength(tag, font=tag_font)
    draw.text((WINDOW_W / 2 - tw / 2, BOTTOM_TAB_H + 5), tag, font=tag_font, fill=TEXT_DIM)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _fit_frame(frame, target_w, target_h):
    """Resizes frame to fit inside target_w x target_h, preserving aspect ratio, letterboxed in black."""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off, y_off = (target_w - nw) // 2, (target_h - nh) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    return canvas


def _render_badge(text, color, font):
    """Small solid pill badge ('Live Analysis' / 'Camera Active'), rendered once and reused every frame."""
    tmp = Image.new('RGB', (1, 1))
    bbox = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 8, 4
    w, h = tw + pad_x * 2, th + pad_y * 2
    img = Image.new('RGB', (w, h), color)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=h // 2, fill=color)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=(10, 14, 26))
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


class UIRenderer:
    """
    Owns the UI font(s), the precomputed video-border masks/badges, and
    the cached top/right/bottom images. Create one instance before the
    camera loop and call render() every frame.
    """
    def __init__(self):
        self.fonts, self.pil_ok = _load_fonts()
        self._top_key,    self._top_img    = None, None
        self._panel_key,  self._panel_img  = None, None
        self._bottom_key, self._bottom_img = None, None
        if self.pil_ok:
            badge_font = self.fonts[('bold', 11)]
            self._live_badge_img   = _render_badge("Live Analysis", GREEN, badge_font)
            self._camera_badge_img = _render_badge("Camera Active", BLUE_ACCENT, badge_font)
        else:
            self._live_badge_img = self._camera_badge_img = None

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

        # Flat two-tone "glow" border (a fainter outer line + a solid
        # inner one). Two things were simplified here for frame-rate
        # reasons, both measured on this machine:
        #   - A true rounded-corner clip (cutting the video's own
        #     corners to background) via a precomputed boolean pixel
        #     mask cost ~3.7ms/frame -- the mask still has to be
        #     scanned over the whole frame every frame even though
        #     it's precomputed. Plain cv2.rectangle here costs ~0.04ms
        #     since it only touches the pixels on the line itself, so
        #     the video's corners stay square instead of rounded.
        #   - A real blurred glow (Gaussian blur every frame) would
        #     cost several more ms for a border that never moves or
        #     changes color; two flat-color rectangles approximate it
        #     instead.
        cv2.rectangle(video, (VIDEO_BORDER_THICK + VIDEO_HALO_THICK, VIDEO_BORDER_THICK + VIDEO_HALO_THICK),
                       (VIDEO_W - 1 - VIDEO_BORDER_THICK - VIDEO_HALO_THICK,
                        VIDEO_H - 1 - VIDEO_BORDER_THICK - VIDEO_HALO_THICK),
                       BLUE_HALO_BGR, VIDEO_HALO_THICK)
        cv2.rectangle(video, (0, 0), (VIDEO_W - 1, VIDEO_H - 1), BLUE_ACCENT_BGR, VIDEO_BORDER_THICK)

        if self._live_badge_img is not None:
            bh, bw = self._live_badge_img.shape[:2]
            video[10:10 + bh, 10:10 + bw] = self._live_badge_img
        if self._camera_badge_img is not None:
            bh, bw = self._camera_badge_img.shape[:2]
            video[VIDEO_H - 10 - bh:VIDEO_H - 10, 10:10 + bw] = self._camera_badge_img

        if state.get('multiple_faces'):
            cv2.putText(video, "Multiple faces detected - using the largest", (36, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, AMBER_BGR, 2)
        if state['analyzing'] and state['near_edge']:
            cv2.putText(video, "Move to center", (36, VIDEO_H - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, AMBER_BGR, 2)
        if state['cnn_busy']:
            pulse = (math.sin(time.time() * 6) + 1) / 2
            radius = int(4 + pulse * 3)
            cv2.circle(video, (VIDEO_W - 40, 40), radius, BLUE_ACCENT_BGR, -1)
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
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 1)
    mic_on = state['voice_status'] in ('on', 'listening')
    cv2.putText(frame, "[MIC ON]" if mic_on else "[MIC OFF]", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if mic_on else (100, 100, 100), 1)
    if state.get('rule_based_mode'):
        cv2.putText(frame, "[RULE-BASED MODE]", (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
    if state.get('multiple_faces'):
        cv2.putText(frame, "Multiple faces - using largest", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    panel_x = w - 230
    cv2.rectangle(frame, (panel_x, 50), (w, h - 50), (20, 20, 20), -1)
    y = 75
    cv2.putText(frame, "FACE SHAPE", (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    y += 22
    if not state.get('face_present', True):
        cv2.putText(frame, "No face detected", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1)
    elif state['analyzing']:
        collected, total = state['analysis_progress']
        cv2.putText(frame, f"Analyzing... {collected}/{total}", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
    else:
        fs = state['face_shape']
        cv2.putText(frame, fs or "Detecting...", (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 150) if fs else (100, 100, 100), 2)
        if fs and state['face_shape_confidence'] is not None:
            y += 18
            cv2.putText(frame, f"{state['face_shape_confidence']:.0f}%", (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        if fs and state.get('learned_count', 0) > 0:
            y += 16
            cv2.putText(frame, f"Learned from {state['learned_count']} ratings", (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (74, 173, 176), 1)

    y += 35
    cv2.putText(frame, "HAIRSTYLE", (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    y += 22
    cv2.putText(frame, get_hair_rec_summary(state['gender'], state['face_shape']), (panel_x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    y += 35
    cv2.putText(frame, "SKIN TONE", (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    y += 22
    cv2.putText(frame, state['skin_tone'] or "Detecting...", (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 200, 255) if state['skin_tone'] else (100, 100, 100), 2)
    if state['skin_tone'] and state.get('skin_tone_low_confidence'):
        y += 16
        cv2.putText(frame, "Rough estimate - shadow detected", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, AMBER_BGR, 1)
    if state['skin_tone']:
        y += 16
        cv2.putText(frame, "For best results, remove hats / even lighting", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (142, 148, 157), 1)

    y += 35
    cv2.putText(frame, "OUTFIT COLORS", (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    y += 20
    for part in state['color_rec'].split(', '):
        cv2.putText(frame, part, (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        y += 16

    y += 15
    cv2.putText(frame, "GLASSES", (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    y += 20
    cv2.putText(frame, get_glasses_rec_summary(state['face_shape']), (panel_x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    cv2.rectangle(frame, (0, h - 30), (w - 230, h), (20, 20, 20), -1)
    if state['glasses_names']:
        tag = " * REC" if state['is_recommended'] else ""
        cv2.putText(frame, f"Glasses {state['glasses_idx']+1}/{state['total_glasses']}: "
                            f"{state['glasses_current_name']}{tag}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


def draw_notification(frame, text, h):
    cv2.putText(
        frame, text,
        (10, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 255), 2
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
    cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), BLUE_ACCENT_BGR, 2)

    label = "Hairstyle" if stage == 'hairstyle' else "Glasses"
    cv2.putText(frame, f"Rate this {label} suggestion: {suggestion_name}",
                (x1 + 16, y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
    cv2.putText(frame, "Press 1-5 to rate  *  Esc to skip",
                (x1 + 16, y1 + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 184, 190), 1)
    return frame


# ──────────────────────────────────────────
# Gender selection screen: rendered ONCE with PIL (not per-frame —
# nothing on it animates), then just redisplayed while waiting for a
# card click.
# ──────────────────────────────────────────
GENDER_CARD_W    = 576
GENDER_CARD_H    = 340
GENDER_CARD_Y    = 236
GENDER_CARD_GAP  = 48
GENDER_SIDE_MARGIN = (WINDOW_W - 2 * GENDER_CARD_W - GENDER_CARD_GAP) // 2
MALE_CARD_RECT   = (GENDER_SIDE_MARGIN, GENDER_CARD_Y,
                     GENDER_SIDE_MARGIN + GENDER_CARD_W, GENDER_CARD_Y + GENDER_CARD_H)
FEMALE_CARD_RECT = (MALE_CARD_RECT[2] + GENDER_CARD_GAP, GENDER_CARD_Y,
                     MALE_CARD_RECT[2] + GENDER_CARD_GAP + GENDER_CARD_W, GENDER_CARD_Y + GENDER_CARD_H)


def _draw_gender_card(bg_img, draw, fonts, rect, title, subtext, color_top, color_bottom):
    x1, y1, x2, y2 = rect
    w, h = x2 - x1, y2 - y1
    card, mask = _rounded_gradient_card(w, h, color_top, color_bottom, radius=24)
    bg_img.paste(card, (x1, y1), mask)

    cx = x1 + w // 2
    if title == "MALE":
        _male_silhouette(draw, cx, y1 + 34, 1.15, TEXT_WHITE)
    else:
        _female_silhouette(draw, cx, y1 + 34, 1.15, TEXT_WHITE)

    title_font = fonts[('bold', 28)]
    tw = draw.textlength(title, font=title_font)
    ty = y1 + 190
    draw.text((cx - tw / 2, ty), title, font=title_font, fill=TEXT_WHITE)

    sub_font = fonts[('regular', 14)]
    sw = draw.textlength(subtext, font=sub_font)
    draw.text((cx - sw / 2, ty + 40), subtext, font=sub_font, fill=(232, 235, 245))

    _arrow_button(draw, x2 - 42, y2 - 42, 22, (255, 255, 255), color_bottom)


def _render_gender_screen(fonts):
    bg = _glow_background(WINDOW_W, WINDOW_H)
    draw = ImageDraw.Draw(bg)
    cx = WINDOW_W // 2

    _sparkle(draw, cx, 46, 12, BLUE_1)

    title = "AI Grooming Assistant"
    title_font = fonts[('bold', 38)]
    tw = draw.textlength(title, font=title_font)
    draw.text((cx - tw / 2, 66), title, font=title_font, fill=TEXT_WHITE)
    line_y = 66 + 26
    _gradient_line(draw, cx - tw / 2 - 120, cx - tw / 2 - 20, line_y, BG_APP, BLUE_1, thickness=2)
    _gradient_line(draw, cx + tw / 2 + 20, cx + tw / 2 + 120, line_y, PURPLE_1, BG_APP, thickness=2)

    subtitle = "Discover Your Best Look - Powered by AI"
    sf = fonts[('regular', 18)]
    sw = draw.textlength(subtitle, font=sf)
    draw.text((cx - sw / 2, 120), subtitle, font=sf, fill=TEXT_DIM)

    desc = "Get personalized style recommendations based on your face shape and skin tone"
    df = fonts[('regular', 13)]
    dw = draw.textlength(desc, font=df)
    draw.text((cx - dw / 2, 150), desc, font=df, fill=TEXT_DIM)

    _draw_gender_card(bg, draw, fonts, MALE_CARD_RECT, "MALE",
                       "Find your perfect style and look", BLUE_1, BLUE_2)
    _draw_gender_card(bg, draw, fonts, FEMALE_CARD_RECT, "FEMALE",
                       "Enhance your style with AI", PURPLE_1, PURPLE_2)

    features = [
        ("Face Shape Analysis", _face_outline_icon, BLUE_1),
        ("Skin Tone Detection", _droplet_icon, PURPLE_1),
        ("Smart Recommendations", _sparkle, BLUE_1),
        ("AR Visualization & Voice Assistant", _glasses_icon, PURPLE_1),
    ]
    fy = GENDER_CARD_Y + GENDER_CARD_H + 46
    cell_w = WINDOW_W // 4
    ffont = fonts[('regular', 12)]
    for i, (label, icon_fn, color) in enumerate(features):
        fx = cell_w * i + cell_w // 2
        draw.ellipse([fx - 26, fy - 26, fx + 26, fy + 26], outline=color, width=2)
        icon_fn(draw, fx, fy, 14, color)
        lines = _wrap_text(draw, label, ffont, cell_w - 40)
        ly = fy + 38
        for line in lines:
            lw = draw.textlength(line, font=ffont)
            draw.text((fx - lw / 2, ly), line, font=ffont, fill=TEXT_DIM)
            ly += 16

    _draw_spaced_text(draw, cx, WINDOW_H - 20, "LOOK GOOD  •  FEEL CONFIDENT",
                       fonts[('regular', 11)], TEXT_DIM, spacing=3)

    return cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)


def _show_gender_selection_fallback():
    """Plain cv2 version, used only if the TrueType font couldn't be loaded at all."""
    selected = [None]
    W, H = 640, 400

    while selected[0] is None:
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        for i in range(H):
            ratio = i / H
            frame[i, :] = [int(20 + ratio * 10), int(20 + ratio * 10), int(40 + ratio * 20)]

        cv2.putText(frame, "AI Grooming Assistant", (80, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 150), 2)
        cv2.putText(frame, "Select Your Gender to Continue", (100, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

        cv2.rectangle(frame, (80, 180), (260, 270), (200, 150, 50), -1)
        cv2.rectangle(frame, (80, 180), (260, 270), (255, 255, 255), 2)
        cv2.putText(frame, "MALE", (130, 235), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.rectangle(frame, (360, 180), (560, 270), (180, 50, 180), -1)
        cv2.rectangle(frame, (360, 180), (560, 270), (255, 255, 255), 2)
        cv2.putText(frame, "FEMALE", (390, 235), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.putText(frame, "Click on your gender", (200, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        cv2.imshow("AI Grooming Assistant", frame)

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if 80 <= x <= 260 and 180 <= y <= 270:
                    selected[0] = 'male'
                elif 360 <= x <= 560 and 180 <= y <= 270:
                    selected[0] = 'female'

        cv2.setMouseCallback("AI Grooming Assistant", on_click)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            selected[0] = 'male'
            break

    cv2.destroyAllWindows()
    print(f"Gender selected: {selected[0]}")
    return selected[0]


def show_gender_selection():
    """
    Show the gender selection screen before the app starts. Rendered
    once with PIL (see _render_gender_screen) since nothing on it
    animates, then just redisplayed while waiting for a card click.
    Returns 'male' or 'female'.
    """
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
        if key == ord('q'):
            selected[0] = 'male'
            break

    cv2.destroyAllWindows()
    print(f"Gender selected: {selected[0]}")
    return selected[0]

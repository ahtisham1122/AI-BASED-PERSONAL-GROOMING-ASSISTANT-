"""
Eraser tool for glasses PNGs: paint out the side arms (temples) so only
the FRONT frame shows in the AR try-on.

    python tools/clean_glasses.py               # every PNG in assets/glasses/processed/
    python tools/clean_glasses.py glasses14     # just these ones
    python tools/clean_glasses.py --selftest    # no window; checks the image logic

Mouse: left-click + drag = erase (round brush).
Keys:  [ / ] brush smaller/bigger   Z undo stroke   R reset image
       S save + next                N skip          Q quit

Saving writes back to assets/glasses/processed/<name>.png at the SAME
size (never cropped/resized). The first save of each image backs up the
untouched PNG to assets/glasses/originals/ (never overwritten).
"""
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from app.paths import ASSETS_DIR

PROCESSED_DIR = str(ASSETS_DIR / 'glasses' / 'processed')
ORIGINALS_DIR = str(ASSETS_DIR / 'glasses' / 'originals')
WINDOW = 'Clean glasses - drag to erase | [ ] brush | Z undo | R reset | S save | N skip | Q quit'

MAX_VIEW_W, MAX_VIEW_H = 1400, 800  # largest on-screen size; big photos are shown scaled down to fit a laptop screen
BRUSH_START, BRUSH_MIN, BRUSH_MAX, BRUSH_STEP = 20, 3, 150, 4  # brush RADIUS in on-screen pixels
CHECKER = 16          # checkerboard square size (on-screen px) -- makes transparent areas visible
MAX_UNDO = 30         # strokes kept for Z; caps memory (~3 MB per stroke on the largest 2053x1360 image)
SOFTEN_KSIZE = 5      # blur size for softening a cut edge: ~2 px of fade each side, enough to hide a hard line without visibly blurring the frame
LOADER_ALPHA_THRESH = 10  # ar_overlay._auto_crop treats alpha > 10 as "visible" when it trims the image
KEEP_BOX_ALPHA = LOADER_ALPHA_THRESH + 1  # 11/255 = ~4% opacity: invisible on screen, but still counted by the trim


# ── Image logic (no window needed; covered by --selftest) ─────────────

def erase_stroke(alpha, p0, p1, radius):
    """Sets alpha to 0 along a round-brush line from p0 to p1 (image pixels)."""
    cv2.line(alpha, p0, p1, 0, thickness=max(1, int(2 * radius)), lineType=cv2.LINE_AA)
    cv2.circle(alpha, p1, max(1, int(radius)), 0, -1, lineType=cv2.LINE_AA)


def soften_erased_edges(alpha, before):
    """Small alpha blur ONLY in a thin band around what was erased, so the cut has no hard line; the rest of the frame is untouched."""
    erased = (before > 0) & (alpha < before)
    if not erased.any():
        return alpha
    band = cv2.dilate(erased.astype(np.uint8), np.ones((SOFTEN_KSIZE, SOFTEN_KSIZE), np.uint8)) > 0
    blurred = cv2.GaussianBlur(alpha, (SOFTEN_KSIZE, SOFTEN_KSIZE), 0)
    return np.where(band, np.minimum(blurred, before), alpha).astype(np.uint8)


def _visible_box(alpha):
    ys, xs = np.where(alpha > LOADER_ALPHA_THRESH)
    return (ys.min(), xs.min(), ys.max(), xs.max()) if len(ys) else None


def keep_crop_box(alpha, before):
    """
    The desktop/web loaders trim each PNG to its visible area and then
    stretch it to face width. If erasing an arm shrank that area, the
    frame would get bigger and shift on the face. Two ~invisible corner
    dots at the ORIGINAL box keep the trim -- so width and position -- identical.
    """
    # ponytail: marker pixels instead of changing the loader; drop them if
    # ar_overlay ever stops auto-cropping (e.g. uses fixed per-image boxes).
    box = _visible_box(before)
    if box is None or _visible_box(alpha) == box:
        return alpha
    y0, x0, y1, x1 = box
    for y, x in ((y0, x0), (y1, x1)):
        alpha[y, x] = max(alpha[y, x], KEEP_BOX_ALPHA)
    return alpha


def backup_original(path):
    """Copies the untouched PNG to originals/ the first time only -- an existing backup is never overwritten."""
    os.makedirs(ORIGINALS_DIR, exist_ok=True)
    dst = os.path.join(ORIGINALS_DIR, os.path.basename(path))
    if not os.path.exists(dst):
        shutil.copy2(path, dst)
    return dst


def finish_and_save(path, rgba, alpha, before_alpha):
    backup_original(path)
    alpha = soften_erased_edges(alpha.copy(), before_alpha)
    rgba[:, :, 3] = keep_crop_box(alpha, before_alpha)
    cv2.imwrite(path, rgba)  # same width x height as loaded: no crop, no resize


# ── Interactive window ────────────────────────────────────────────────

def _checkerboard(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    tile = ((yy // CHECKER + xx // CHECKER) % 2).astype(np.uint8)
    return np.where(tile[..., None] == 0, 205, 160).astype(np.uint8).repeat(3, axis=2)


class Editor:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.before = self.rgba[:, :, 3].copy()
        self.alpha = self.before.copy()  # own contiguous array: cv2 can't draw on the rgba[:, :, 3] slice
        h, w = self.rgba.shape[:2]
        self.scale = min(MAX_VIEW_W / w, MAX_VIEW_H / h, 1.0)
        self.view = (int(w * self.scale), int(h * self.scale))
        self.checker = _checkerboard(self.view[1], self.view[0])
        self.brush = BRUSH_START
        self.undo = []
        self.last = None
        self.mouse = None

    def to_image(self, x, y):
        return int(x / self.scale), int(y / self.scale)

    def on_mouse(self, event, x, y, flags, _):
        self.mouse = (x, y)
        p = self.to_image(x, y)
        r = self.brush / self.scale
        if event == cv2.EVENT_LBUTTONDOWN:
            self.undo = (self.undo + [self.alpha.copy()])[-MAX_UNDO:]
            erase_stroke(self.alpha, p, p, r)
            self.last = p
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON and self.last:
            erase_stroke(self.alpha, self.last, p, r)
            self.last = p
        elif event == cv2.EVENT_LBUTTONUP:
            self.last = None

    def render(self):
        bgr = cv2.resize(self.rgba[:, :, :3], self.view, interpolation=cv2.INTER_AREA)
        a = cv2.resize(self.alpha, self.view, interpolation=cv2.INTER_AREA)[..., None].astype(np.float32) / 255.0
        out = (bgr * a + self.checker * (1 - a)).astype(np.uint8)
        if self.mouse:
            cv2.circle(out, self.mouse, self.brush, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(out, f"{self.name}  brush {self.brush}px  undo {len(self.undo)}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2, cv2.LINE_AA)
        return out

    def handle_key(self, key):
        """Returns 'save' / 'skip' / 'quit', or None to keep editing."""
        if key == ord('['):
            self.brush = max(BRUSH_MIN, self.brush - BRUSH_STEP)
        elif key == ord(']'):
            self.brush = min(BRUSH_MAX, self.brush + BRUSH_STEP)
        elif key in (ord('z'), ord('Z')) and self.undo:
            self.alpha = self.undo.pop()
        elif key in (ord('r'), ord('R')):
            self.alpha = self.before.copy()
            self.undo.clear()
        elif key in (ord('s'), ord('S')):
            return 'save'
        elif key in (ord('n'), ord('N')):
            return 'skip'
        elif key in (ord('q'), ord('Q')):
            return 'quit'
        return None


def edit_one(path):
    ed = Editor(path)
    if ed.rgba is None or ed.rgba.ndim != 3 or ed.rgba.shape[2] != 4:
        print(f"[clean_glasses] {ed.name}: not a PNG with transparency, skipped")
        return 'skip'
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, ed.on_mouse)
    while True:
        cv2.imshow(WINDOW, ed.render())
        key = cv2.waitKey(15) & 0xFF
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return 'quit'  # window closed with the X button
        action = ed.handle_key(key)
        if action == 'save':
            finish_and_save(path, ed.rgba, ed.alpha, ed.before)
            print(f"[clean_glasses] saved {ed.name} (original kept in {ORIGINALS_DIR})")
        if action:
            return action


def pick_files(args):
    if args:
        return [os.path.join(PROCESSED_DIR, a if a.endswith('.png') else a + '.png') for a in args]
    names = [f for f in os.listdir(PROCESSED_DIR) if f.lower().endswith('.png')]
    return [os.path.join(PROCESSED_DIR, f) for f in sorted(names, key=lambda s: (len(s), s))]


def main(args):
    for path in pick_files(args):
        if not os.path.exists(path):
            print(f"[clean_glasses] not found: {path}")
            continue
        if edit_one(path) == 'quit':
            break
    cv2.destroyAllWindows()


def selftest():
    """Checks erase / soften / keep-box / backup on a synthetic image, no window."""
    import tempfile
    global ORIGINALS_DIR
    before = np.zeros((100, 200), np.uint8)
    before[40:60, 20:180] = 255   # the "front frame"
    before[10:40, 170:190] = 255  # an "arm" sticking up past the frame's top
    alpha = before.copy()
    erase_stroke(alpha, (180, 5), (180, 39), 12)
    assert alpha[20, 180] == 0 and alpha[50, 100] == 255, "erase hit the wrong place"
    soft = soften_erased_edges(alpha.copy(), before)
    assert soft[50, 100] == 255, "softening touched pixels far from the cut"
    assert (soft <= before).all(), "softening brought back alpha that wasn't there"
    kept = keep_crop_box(soft.copy(), before)
    assert _visible_box(kept) == _visible_box(before), "trim box changed -> glasses would move/resize"
    changed = kept != soft
    assert changed.sum() == 2 and (kept[changed] == KEEP_BOX_ALPHA).all(), "expected exactly 2 near-invisible marker pixels"
    with tempfile.TemporaryDirectory() as tmp:
        ORIGINALS_DIR = os.path.join(tmp, 'originals')
        src = os.path.join(tmp, 'g.png')
        open(src, 'wb').write(b'first')
        backup_original(src)
        open(src, 'wb').write(b'second')
        backup_original(src)
        assert open(os.path.join(ORIGINALS_DIR, 'g.png'), 'rb').read() == b'first', "backup was overwritten"
    print("[clean_glasses] selftest passed")


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        selftest()
    else:
        main(sys.argv[1:])

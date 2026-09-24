"""
One-time glasses asset prep. Run by hand whenever new images are
added to assets/glasses/ -- NOT called by the live app.

For every image in assets/glasses/ (PNG/JPG/JPEG/WEBP):
  - Already has a real transparent background -> copied as-is.
  - Otherwise, background removed with rembg (first choice; slow is
    fine, this is one-time) or, if that fails, a corner flood-fill
    fallback with a softened edge.
  - Lens interiors (still background-colored after removal, since the
    frame blocks a corner flood-fill from ever reaching them) are made
    mostly see-through: ~20% visible for a clear lens, ~85% for a dark
    one (sunglasses).
  - Auto-cropped to visible content and saved as a PNG in
    assets/glasses/processed/ (originals are never touched).

Also writes assets/glasses/glasses_preview.png, a grid of every
processed pair on a dark background with filenames underneath, so you
can eyeball the whole batch at once.
"""
import cv2
import numpy as np
import os

SOURCE_DIR = os.path.join('assets', 'glasses')
OUT_DIR = os.path.join(SOURCE_DIR, 'processed')
PREVIEW_PATH = os.path.join(SOURCE_DIR, 'glasses_preview.png')
VALID_EXTS = ('.png', '.jpg', '.jpeg', '.webp')

CLEAR_LENS_ALPHA = 0.20    # "mostly see-through" clear glass
DARK_LENS_ALPHA  = 0.85    # sunglasses stay mostly opaque/tinted
DARK_LUMA_THRESH = 90      # below this brightness (0-255), an enclosed region counts as a dark/sunglasses lens rather than clear
BG_COLOR_TOLERANCE = 60    # how close (sum of per-channel abs diff) a pixel must be to the sampled corner color to count as "background-like"
FLOODFILL_TOLERANCE = 18   # fallback flood-fill color tolerance per channel
MIN_HOLE_AREA = 40         # ignore tiny enclosed specks (compression noise, tiny highlights) -- not real lens holes
MIN_HOLE_FILL_RATIO = 0.35 # an enclosed region must fill at least this fraction of its own bounding box to count as a lens, not thin frame material


def _has_real_alpha(bgra):
    """Same check as ar_overlay.remove_white_bg: some pixels meaningfully transparent, not just a fully-opaque channel tacked on by the exporter."""
    if bgra is None or bgra.shape[2] != 4:
        return False
    alpha = bgra[:, :, 3]
    return alpha.min() < 250 and (alpha < 250).mean() > 0.01


def _remove_bg_rembg(bgr):
    from rembg import remove
    from PIL import Image
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = remove(Image.fromarray(rgb)).convert('RGBA')
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGBA2BGRA)


def _remove_bg_floodfill(bgr, tolerance=FLOODFILL_TOLERANCE):
    """Fallback: flood-fill from all 4 corners with a color tolerance, then soften the edge so there's no hard white outline."""
    h, w = bgr.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # FLOODFILL_FIXED_RANGE matters here: without it, cv2.floodFill
    # checks tolerance neighbor-to-neighbor rather than against the
    # original seed color, so it "walks" through a smooth gradient
    # (a soft drop-shadow, an anti-aliased edge) many small steps at a
    # time and leaks straight through into the frame -- confirmed by
    # testing against a real photo, where it ate the entire frame.
    flood_flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    work = bgr.copy()
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        cv2.floodFill(work, mask, seed, 0, (tolerance,) * 3, (tolerance,) * 3, flood_flags)
    bg_mask = mask[1:-1, 1:-1]
    alpha = np.where(bg_mask > 0, 0, 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)  # soften 1-2px so there's no hard white outline
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return rgba


def _enclosed_components(mask_u8, w, h):
    """Connected components of mask_u8 that don't touch the image border (real background always does) and are chunky enough to be a lens, not thin frame material."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    for label in range(1, num):
        x, y, cw, ch, area = stats[label]
        if area < MIN_HOLE_AREA:
            continue
        if x <= 0 or y <= 0 or x + cw >= w or y + ch >= h:
            continue
        if area / float(cw * ch) < MIN_HOLE_FILL_RATIO:
            continue
        yield labels == label


def _apply_lens_transparency(rgba, bg_color):
    """
    Finds regions enclosed by the frame that background removal
    couldn't reach (a corner flood-fill can't get past the frame into
    the lens interior), and makes them mostly see-through.

    bg_color must be sampled from the ORIGINAL image's corners, before
    background removal -- rembg (and some other removers) zero out the
    RGB channels wherever alpha becomes 0, so sampling color from the
    post-removal image's own corners reads black instead of the true
    background color, and a still-opaque white lens interior then
    never matches it (confirmed: this silently left every rembg-
    processed lens fully solid instead of see-through).
    """
    h, w = rgba.shape[:2]
    bgr = rgba[:, :, :3].astype(np.int32)
    alpha = rgba[:, :, 3].astype(np.float32)

    bg_like = (np.abs(bgr - bg_color).sum(axis=2) < BG_COLOR_TOLERANCE)

    out_alpha = alpha.copy()

    # Clear lenses: still opaque (background removal never reached them) and still background-colored.
    clear_candidates = ((alpha > 10) & bg_like).astype(np.uint8)
    for comp in _enclosed_components(clear_candidates, w, h):
        out_alpha[comp] = 255 * CLEAR_LENS_ALPHA

    # Dark lenses: opaque, NOT background-colored (they have their own tint), and dark.
    luma = 0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]
    dark_candidates = ((alpha > 10) & (~bg_like) & (luma < DARK_LUMA_THRESH)).astype(np.uint8)
    for comp in _enclosed_components(dark_candidates, w, h):
        out_alpha[comp] = 255 * DARK_LENS_ALPHA

    out = rgba.copy()
    out[:, :, 3] = out_alpha.astype(np.uint8)
    return out


def _auto_crop(rgba, alpha_thresh=10, pad=2):
    alpha = rgba[:, :, 3]
    mask = alpha > alpha_thresh
    if not mask.any():
        return rgba
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - pad), min(rgba.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(rgba.shape[1], xs.max() + 1 + pad)
    return rgba[y0:y1, x0:x1]


def process_one(path):
    """Returns (rgba_result_or_None, method) -- method in 'copied' / 'rembg' / 'floodfill' / 'failed'."""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None, 'failed'
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

    if raw.shape[2] == 4 and _has_real_alpha(raw):
        return raw, 'copied'

    bgr = raw[:, :, :3] if raw.shape[2] == 4 else raw
    oh, ow = bgr.shape[:2]
    orig_corners = np.array(
        [bgr[0, 0], bgr[0, ow - 1], bgr[oh - 1, 0], bgr[oh - 1, ow - 1]], dtype=np.float64)
    bg_color = np.median(orig_corners, axis=0)

    try:
        rgba = _remove_bg_rembg(bgr)
        method = 'rembg'
    except Exception as e:
        print(f"    rembg unavailable/failed ({e}); using flood-fill fallback")
        rgba = _remove_bg_floodfill(bgr)
        method = 'floodfill'

    rgba = _apply_lens_transparency(rgba, bg_color)
    rgba = _auto_crop(rgba)
    return rgba, method


def build_preview_sheet(entries):
    """entries: [(out_filename, rgba), ...]. Grid preview on a dark background, filename under each."""
    if not entries:
        return
    cell, label_h = 220, 26
    cols = min(4, len(entries))
    rows = (len(entries) + cols - 1) // cols
    dark_bg = (20, 14, 10)
    sheet = np.full((rows * (cell + label_h), cols * cell, 3), dark_bg, dtype=np.uint8)

    for i, (name, rgba) in enumerate(entries):
        r, c = divmod(i, cols)
        ih, iw = rgba.shape[:2]
        scale = min((cell - 20) / iw, (cell - 20) / ih, 1.0) if iw and ih else 1.0
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        thumb = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA)  # only ever shrinking here
        alpha = thumb[:, :, 3:4].astype(np.float32) / 255.0
        bg_tile = np.full((nh, nw, 3), dark_bg, dtype=np.uint8)
        blended = (alpha * thumb[:, :, :3] + (1 - alpha) * bg_tile).astype(np.uint8)

        x0 = c * cell + (cell - nw) // 2
        y0 = r * (cell + label_h) + (cell - nh) // 2
        sheet[y0:y0 + nh, x0:x0 + nw] = blended

        label = name if len(name) <= 24 else name[:21] + '...'
        (tw, _th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tx = c * cell + (cell - tw) // 2
        ty = r * (cell + label_h) + cell + 18
        cv2.putText(sheet, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 224, 230), 1, cv2.LINE_AA)

    cv2.imwrite(PREVIEW_PATH, sheet)
    print(f"\nSaved preview sheet: {PREVIEW_PATH}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    preview_name = os.path.basename(PREVIEW_PATH)
    files = sorted(
        f for f in os.listdir(SOURCE_DIR)
        if os.path.isfile(os.path.join(SOURCE_DIR, f))
        and os.path.splitext(f)[1].lower() in VALID_EXTS
        and f != preview_name
    )

    results = {'copied': [], 'rembg': [], 'floodfill': [], 'failed': []}
    entries = []

    for f in files:
        path = os.path.join(SOURCE_DIR, f)
        print(f"Processing {f}...")
        rgba, method = process_one(path)
        results[method].append(f)
        if rgba is None:
            print(f"    FAILED to read/process {f}")
            continue
        out_name = os.path.splitext(f)[0] + '.png'
        out_path = os.path.join(OUT_DIR, out_name)
        cv2.imwrite(out_path, rgba)
        entries.append((out_name, rgba))
        print(f"    -> {out_path}  ({method}, {rgba.shape[1]}x{rgba.shape[0]})")

    build_preview_sheet(entries)

    print("\n=== Summary ===")
    print(f"Processed: {len(entries)} / {len(files)}")
    print(f"Copied as-is (already transparent): {results['copied']}")
    print(f"Background removed via rembg: {results['rembg']}")
    print(f"Background removed via flood-fill fallback: {results['floodfill']}")
    if results['failed']:
        print(f"FAILED to process: {results['failed']}")


if __name__ == "__main__":
    main()

"""
ARCHIVED: one-off visual QA tool from when remove_white_bg() (now in
ar_overlay.py) was reworked from a hard threshold to soft-threshold +
edge-blur. That change already shipped; this script and its
debug_output/ were only needed to review it before committing.

One-off visual QA tool: compares the OLD hard-threshold background
removal against the NEW soft-threshold + edge-blur version in
main.py, so you can eyeball edge quality before committing.

Also dumps how your 4 real glasses PNGs look today, as-is, so you
can see the existing halo (which comes from color baked into the
PNGs' own semi-transparent edge pixels, not from remove_white_bg —
see the debug notes printed at the end).

Run: python compare_glasses_bg.py
Output: debug_output/<glasses-name>/*.png
Delete this file and debug_output/ once you're done checking.
"""
import os
import cv2
import numpy as np

from main import remove_white_bg  # the new implementation

GLASSES_FOLDER = 'assets/glasses'
OUT_FOLDER     = 'debug_output'
# Mid-tone background makes white/gray edge halos obvious.
CHECK_BG = np.array([120, 140, 170], dtype=np.uint8)  # BGR, skin-ish tan


def old_remove_white_bg(img, threshold=230):
    """Exact copy of the original hard-threshold + 3x3 blur logic,
    kept only so we have something to compare the new one against."""
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    mask = (
        (rgba[:, :, 0] > threshold) &
        (rgba[:, :, 1] > threshold) &
        (rgba[:, :, 2] > threshold)
    )
    rgba[mask, 3] = 0
    alpha = rgba[:, :, 3].astype(float)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    rgba[:, :, 3] = alpha.astype(np.uint8)
    return rgba


def flatten_on_white(rgba):
    """Turns an existing transparent PNG back into a plain white-
    background BGR image, simulating a fresh unprocessed asset —
    this is the input shape remove_white_bg is actually meant for."""
    alpha = rgba[:, :, 3:4].astype(float) / 255.0
    white = np.full(rgba[:, :, :3].shape, 255, dtype=float)
    flat = (
        alpha * rgba[:, :, :3].astype(float) +
        (1 - alpha) * white
    ).astype(np.uint8)
    return flat


def composite_on(rgba, bg_color):
    """Blends an RGBA image onto a solid background color."""
    h, w = rgba.shape[:2]
    bg = np.full((h, w, 3), bg_color, dtype=np.uint8)
    alpha = rgba[:, :, 3:4].astype(float) / 255.0
    out = (
        alpha * rgba[:, :, :3].astype(float) +
        (1 - alpha) * bg.astype(float)
    ).astype(np.uint8)
    return out


def main():
    os.makedirs(OUT_FOLDER, exist_ok=True)
    files = sorted(
        f for f in os.listdir(GLASSES_FOLDER)
        if f.lower().endswith('.png')
    )

    for f in files:
        name = f.replace('.png', '')
        img = cv2.imread(
            os.path.join(GLASSES_FOLDER, f), cv2.IMREAD_UNCHANGED
        )
        if img is None or img.shape[2] != 4:
            print(f"skip {f}: no alpha to work with")
            continue

        out_dir = os.path.join(OUT_FOLDER, name)
        os.makedirs(out_dir, exist_ok=True)

        # 1) The halo as it looks TODAY, in the real app, from the
        #    asset's own baked-in edge pixels.
        cv2.imwrite(
            os.path.join(out_dir, '1_current_asset_on_bg.png'),
            composite_on(img, CHECK_BG)
        )

        # 2) Old vs new algorithm, run on a flattened (no-alpha)
        #    version of the same image — this is the scenario
        #    remove_white_bg is actually for (a brand-new PNG with
        #    a plain white background and no alpha yet).
        flat = flatten_on_white(img)
        old_rgba = old_remove_white_bg(flat.copy())
        new_rgba = remove_white_bg(flat.copy())

        cv2.imwrite(
            os.path.join(out_dir, '2_flattened_input.png'), flat
        )
        cv2.imwrite(
            os.path.join(out_dir, '3_old_algorithm_on_bg.png'),
            composite_on(old_rgba, CHECK_BG)
        )
        cv2.imwrite(
            os.path.join(out_dir, '4_new_algorithm_on_bg.png'),
            composite_on(new_rgba, CHECK_BG)
        )
        print(f"{name}: saved comparison to {out_dir}/")

    print(
        f"\nDone. In each debug_output/<glasses>/ folder:\n"
        f"  1_current_asset_on_bg.png   = what you see in the app today\n"
        f"  3_old_algorithm_on_bg.png   = old threshold logic's result\n"
        f"  4_new_algorithm_on_bg.png   = new soft-threshold logic's result\n"
        f"Compare 3 vs 4 for the algorithm improvement. Note that "
        f"1 won't change from this update — see the script's docstring."
    )


if __name__ == '__main__':
    main()

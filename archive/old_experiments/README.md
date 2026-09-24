# Archived experiments

Everything here is superseded and unused by the live app (`main.py` and
the modules it imports never read from this folder). Kept for the FYP
report/history, not for running.

- **`build_dataset_from_kaggle.py`** — built the ratio CSV below. Superseded by the MobileNetV2 CNN (`train_face_shape_cnn_v2.ipynb`), which measures accuracy in the 80%+ range this ratio approach couldn't reach.
- **`face_shape_training_data.csv`** — the ratio-based training data (`hr`/`fr`/`jr`/label per photo) produced by the script above, built with the *old* aspect-ratio-buggy version of the ratio math. Superseded by the CNN; not regenerated after the ratio-math fix since the CNN replaced this approach entirely.
- **`train_face_shape_model.ipynb`** — trained a Random Forest on the CSV above (~45% accuracy). Superseded by `train_face_shape_cnn_v2.ipynb`, which trains the MobileNetV2 CNN actually used by the live app.
- **`train_face_shape_cnn.ipynb`** — the first CNN notebook attempt. Superseded by `train_face_shape_cnn_v2.ipynb`, which fixed framing/preprocessing issues found by testing this version's output; the live model files were trained with v2, not this one.
- **`compare_glasses_bg.py`** + **`debug_output/`** — one-off visual QA tool used while reworking `remove_white_bg()` (now in `ar_overlay.py`) from a hard threshold to soft-threshold + edge-blur. That change already shipped; these were only needed to review it before committing.

# AI-Based Personal Grooming Assistant

Real-time camera app that detects face shape and skin tone and gives style recommendations.

## Source of truth

`main.py` is the single source of truth. It is the real app.

`dev_scripts_old/` is an archive of old test scripts (`face_shape.py`, `skin_tone_old.py`, `ar_glasses.py`, `face_detection.py`, `sonu.py`, `test_cam.py`, `mic_test.py`). They are unused and kept only for reference. `skin_tone.py` in that folder was renamed to `skin_tone_old.py` to avoid confusion with the real top-level `skin_tone.py` module. Never edit them and never bring their logic back into the real modules — if a real module needs similar logic, write it fresh there.

## Real files

`main.py` is now just the camera loop; everything else is split into modules it calls into:

- `main.py` — camera loop only: opens the camera (with retry/"camera lost" handling if the feed drops), runs the per-frame pipeline, draws the UI, handles keys, logs unexpected per-frame errors to `app.log` instead of crashing.
- `face_tracking.py` — MediaPipe FaceLandmarker setup/download (`face_landmarker.task`), per-frame landmark extraction (picks the largest face if more than one is in frame), `get_head_yaw`, and the two smoothing helpers (`OneEuroFilter` for continuous values, `StableDetector` for debouncing discrete labels).
- `face_shape_model.py` — face shape classification: loads the trained CNN (`face_shape_cnn.tflite`, falling back to `face_shape_cnn.keras`) and runs it on a background thread (`CNNFaceShapeClassifier`) so the camera loop never blocks on inference; falls back to the geometric rule-based classifier (`compute_face_ratios` / `classify_face_shape`) if neither model file loads. Prints which mode is active at startup (and the UI shows a "Rule-based mode" badge). Also owns the BlazeFace detector (`blaze_face_short_range.tflite`) used to crop faces the same way the training data was cropped.
- `skin_tone.py` — skin tone sampling and classification; skips samples that are too dark/bright to be reliable.
- `recommendations.py` — hairstyle/grooming/glasses/outfit-color tables and their lookup functions; re-ranks hairstyle and glasses suggestions using `feedback.py`'s stored ratings.
- `feedback.py` — stores 1-5 user ratings on suggestions in `feedback.json` and computes per-suggestion average ratings for `recommendations.py` to re-rank by. Off by default until the first rating.
- `ar_overlay.py` — glasses asset loading (`assets/glasses/`), background removal, and the overlay/blend/rotation logic.
- `ui.py` — the gender selection screen, the full PIL-rendered card panel (`UIRenderer`/`build_ui_state`), the rating prompt, and notification drawing. Falls back to a plain cv2 layout if its bundled font (`assets/fonts/Inter-Variable.ttf`) can't be loaded.
- `voice_assistant.py` — voice command handling; detects at startup whether a microphone is even available.
- `face_shape_cnn.tflite` / `face_shape_cnn.keras` — trained face shape CNN (tflite tried first, keras as fallback). `class_names.json` holds the label order the model outputs. **Never train or re-export these locally** — see below.
- `train_face_shape_cnn_v2.ipynb` — the Colab notebook that trained the CNN files above. This is the current one; older training notebooks/scripts are in `archive_old_experiments/` (see below).
- `face_landmarker.task` / `blaze_face_short_range.tflite` — MediaPipe model files. Auto-downloaded if missing; don't need to commit or hand-manage them.
- `feedback.json` — the ratings store `feedback.py` reads/writes. Doesn't exist until the first rating is given.
- `app.log` — unexpected per-frame exceptions land here (with traceback) instead of crashing the app. Should normally be empty/absent.
- `archive_old_experiments/` — superseded by the CNN approach (old ratio-based Random Forest pipeline, an earlier CNN notebook, and an old background-removal QA tool). Never used by the live app; see its own README.md for what replaced each file.
- `face_shape_dataset/` — the ~1.4GB Kaggle training image set. **Lives one level up, at `../face_shape_dataset` (i.e. outside this project folder)**, not needed for the app to run day-to-day. The only runtime touchpoint is `face_shape_model.py`'s optional debug-crop side-by-side comparison (the `D` key) — it degrades gracefully (just skips the comparison) if the folder isn't there. If you need to rebuild training data, look in `archive_old_experiments/` for the old dataset-building script or point a new one at `../face_shape_dataset`.

## No GPU on this laptop

This machine has no GPU. Never train models locally — this includes the face shape CNN. All model training (the ratio-based Random Forest and the MobileNetV2 CNN) happens on Google Colab; download the trained file(s) afterward (`face_shape_cnn.keras`, `face_shape_cnn.tflite`, `class_names.json`) and place them in the project folder. If the CNN ever needs retraining, go back to Colab, not this machine.

## Camera testing

Camera features need a real webcam. Claude Code cannot run or see `cv2.imshow()` windows itself. After making a camera-related code change, don't claim it works — instead say exactly what to run and what to look for when testing it yourself (e.g. "run `python main.py`, look for X on screen, press Y to check Z").

## Code style

- Keep functions small.
- Any new "magic number" threshold (e.g. a ratio, pixel count, confidence cutoff) must have a comment explaining what it means and why that value — this is for an FYP defense, so it needs to be explainable in plain terms.

## Before making changes

Before editing code, explain in plain simple English what you're about to change and why. No jargon.

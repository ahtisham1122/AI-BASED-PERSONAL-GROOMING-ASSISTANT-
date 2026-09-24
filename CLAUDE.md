# AI-Based Personal Grooming Assistant

Real-time camera app that detects face shape and skin tone and gives style recommendations.

## Source of truth

`main.py` is the single source of truth. It is the real app.

`archive/dev_scripts_old/` is an archive of old test scripts (`face_shape.py`, `skin_tone_old.py`, `ar_glasses.py`, `face_detection.py`, `sonu.py`, `test_cam.py`, `mic_test.py`). They are unused and kept only for reference. `skin_tone.py` in that folder was renamed to `skin_tone_old.py` to avoid confusion with the real top-level `skin_tone.py` module. Never edit them and never bring their logic back into the real modules — if a real module needs similar logic, write it fresh there.

## Real files

`main.py` (project root) is now just the camera loop; everything else is split into modules in `app/` it calls into. Every file path is built from `app/paths.py` (project-folder based), never from the terminal's current folder:

- `main.py` — camera loop only: opens the camera (with retry/"camera lost" handling if the feed drops), runs the per-frame pipeline, draws the UI, handles keys, logs unexpected per-frame errors to `data/app.log` instead of crashing.
- `app/paths.py` — `ROOT`, `MODELS_DIR`, `ASSETS_DIR`, `DATA_DIR` (runtime output) and `DATASET_DIR` (`../face_shape_dataset`).
- `app/face_tracking.py` — MediaPipe FaceLandmarker setup/download (`models/face_landmarker.task`), per-frame landmark extraction (picks the largest face if more than one is in frame), `get_head_yaw`, and the two smoothing helpers (`OneEuroFilter` for continuous values, `StableDetector` for debouncing discrete labels).
- `app/face_shape_model.py` — face shape classification: loads the trained CNN (`models/face_shape_cnn.tflite`, falling back to `models/face_shape_cnn.keras`) and runs it on a background thread (`CNNFaceShapeClassifier`) so the camera loop never blocks on inference; falls back to the geometric rule-based classifier (`compute_face_ratios` / `classify_face_shape`) if neither model file loads. Prints which mode is active at startup (and the UI shows a "Rule-based mode" badge). Also owns the BlazeFace detector (`models/blaze_face_short_range.tflite`) used to crop faces the same way the training data was cropped.
- `app/skin_tone.py` — skin tone sampling and classification; skips samples that are too dark/bright to be reliable.
- `app/recommendations.py` — hairstyle/grooming/glasses/outfit-color tables and their lookup functions; re-ranks hairstyle and glasses suggestions using `feedback.py`'s stored ratings.
- `app/feedback.py` — stores 1-5 user ratings on suggestions in `data/feedback.json` and computes per-suggestion average ratings for `recommendations.py` to re-rank by. Off by default until the first rating.
- `app/ar_overlay.py` — glasses asset loading (`assets/glasses/processed/`, made by `tools/prepare_glasses.py`), background removal, and the overlay/blend/rotation logic.
- `app/ui.py` — the gender selection screen, the full PIL-rendered card panel (`UIRenderer`/`build_ui_state`), the rating prompt, and notification drawing. Falls back to a plain cv2 layout if its bundled font (Poppins, `assets/fonts/`) can't be loaded.
- `app/voice_assistant.py` — voice command handling; detects at startup whether a microphone is even available.
- `app/performance_logger.py` — FPS/inference/memory logger used by main.py; writes to `docs/performance/`.
- `models/face_shape_cnn.tflite` / `models/face_shape_cnn.keras` — trained face shape CNN (tflite tried first, keras as fallback). `models/class_names.json` holds the label order the model outputs. **Never train or re-export these locally** — see below.
- `training/train_face_shape_cnn_v2.ipynb` — the Colab notebook that trained the CNN files above. This is the current one; older training notebooks/scripts are in `archive/old_experiments/` (see below).
- `models/face_landmarker.task` / `models/blaze_face_short_range.tflite` — MediaPipe model files. Auto-downloaded if missing; don't need to commit or hand-manage them.
- `data/feedback.json` — the ratings store `feedback.py` reads/writes. Doesn't exist until the first rating is given.
- `data/app.log` — unexpected per-frame exceptions land here (with traceback) instead of crashing the app. Should normally be empty/absent.
- `webapp/` — Flask backend (`backend/backend_api.py`, port 5001, imports `app.*`) and static frontend (`frontend/`, port 8000).
- `tests/` — plain assert scripts, run with `python tests/<file>.py`.
- `tools/` — one-off dev helpers (glasses preprocessing, UI benchmark/preview, recommendations table).
- `docs/` — recommendations table and performance/stress-test results.
- `archive/` — unused code kept for reference, never used by the live app: `old_experiments/` (the old ratio-based Random Forest pipeline, an earlier CNN notebook and an old background-removal QA tool, all superseded by the CNN; its README.md says what replaced each file), `dev_scripts_old/`, the old Streamlit `app_web.py` + `requirements_web.txt`, and the unused Inter font.
- `face_shape_dataset/` — the ~1.4GB Kaggle training image set. **Lives one level up, at `../face_shape_dataset` (i.e. outside this project folder)**, not needed for the app to run day-to-day. The only runtime touchpoint is `app/face_shape_model.py`'s optional debug-crop side-by-side comparison (the `D` key) — it degrades gracefully (just skips the comparison) if the folder isn't there. If you need to rebuild training data, look in `archive/old_experiments/` for the old dataset-building script or point a new one at `../face_shape_dataset`.

## No GPU on this laptop

This machine has no GPU. Never train models locally — this includes the face shape CNN. All model training (the ratio-based Random Forest and the MobileNetV2 CNN) happens on Google Colab; download the trained file(s) afterward (`face_shape_cnn.keras`, `face_shape_cnn.tflite`, `class_names.json`) and place them in `models/`. If the CNN ever needs retraining, go back to Colab, not this machine.

## Camera testing

Camera features need a real webcam. Claude Code cannot run or see `cv2.imshow()` windows itself. After making a camera-related code change, don't claim it works — instead say exactly what to run and what to look for when testing it yourself (e.g. "run `python main.py`, look for X on screen, press Y to check Z").

## Code style

- Keep functions small.
- Any new "magic number" threshold (e.g. a ratio, pixel count, confidence cutoff) must have a comment explaining what it means and why that value — this is for an FYP defense, so it needs to be explainable in plain terms.

## Before making changes

Before editing code, explain in plain simple English what you're about to change and why. No jargon.

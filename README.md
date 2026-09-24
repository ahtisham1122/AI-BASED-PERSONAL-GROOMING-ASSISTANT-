# AI-Based Personal Grooming Assistant

A real-time camera app that detects your **face shape** (a MobileNetV2 CNN, with a rule-based fallback) and **skin tone** from a webcam. It then recommends hairstyles, grooming, glasses and outfit colours, lets you try glasses on in AR, and takes voice commands and 1–5 star ratings, which it uses to re-rank its suggestions. A web version does the same analysis on an uploaded photo.

## Folder structure

```
fyp/
├── main.py            # Desktop app entry point: the camera loop (run this)
├── app/               # Desktop app modules: face tracking, face shape, skin tone, recommendations,
│                      #   feedback, AR glasses, UI, voice, performance logger, paths.py (all folder paths)
├── models/            # Trained face-shape CNN (.tflite/.keras), class_names.json, MediaPipe models
├── assets/            # Fonts (Poppins) and glasses images (processed/ is what the app loads)
├── webapp/
│   ├── backend/       # Flask API (port 5001) — reuses the app/ modules on one photo
│   └── frontend/      # Static HTML/CSS/JS site (port 8000) that calls the backend
├── training/          # Colab training notebook + script to check model accuracy on the dataset
├── tests/             # Automated checks (plain Python assert scripts)
├── tools/             # Dev helpers: glasses image preprocessing, UI benchmark/preview, recommendations table
├── docs/              # Recommendations table, performance and stress-test results for the report
├── archive/           # Old experiments and scripts no longer used (kept for reference only)
├── data/              # Created at run time: app.log, feedback.json, snapshots/, debug/ (not in git)
├── requirements.txt   # Desktop app dependencies
└── CLAUDE.md          # Notes for the Claude Code assistant
```

The training dataset (`face_shape_dataset/`, ~1.4 GB) is kept **outside** this folder, at `../face_shape_dataset`. The app runs without it; only the model-accuracy script, some tools and the `D`-key debug comparison use it.

## Install

Needs Python 3.11.

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt                   # desktop app
pip install -r webapp/backend/requirements.txt    # web backend (Flask)
```

## Run the desktop app

```bash
python main.py
```

Pick a gender on the start screen. Live analysis then starts from the webcam. You can run it from any folder, e.g. `python D:\fyp\main.py`, because every path is built from the project folder.

## Run the web version

Use two terminals, both from the project folder:

```bash
# Terminal 1 — backend API on http://localhost:5001
python webapp/backend/backend_api.py

# Terminal 2 — frontend on http://localhost:8000
cd webapp/frontend
python -m http.server 8000
```

Then open <http://localhost:8000> in a browser.

## Run the tests

```bash
python tests/test_shape_rule.py            # rule-based face shape classifier
python tests/test_voice_matches_panel.py   # voice replies match the panel's labels, command matching
python tests/test_backend_api.py           # Flask API: error handling + one real photo (needs the dataset)
```

Each prints `all passed`, or stops with an `AssertionError` that explains what failed.

## Trained model files

Put these in **`models/`**:

| File | What it is |
|---|---|
| `face_shape_cnn.tflite` | Face-shape CNN (loaded first) |
| `face_shape_cnn.keras` | Same model, used if the .tflite fails to load |
| `class_names.json` | Label order the model outputs |
| `face_landmarker.task`, `blaze_face_short_range.tflite` | MediaPipe models, **downloaded automatically** on first run |

The CNN is trained on Google Colab with `training/train_face_shape_cnn_v2.ipynb`, not on this laptop (it has no GPU). If the CNN files are missing, the app falls back to the rule-based classifier and shows a "Rule-based mode" badge.

## How to clean a new glasses image

Product photos often show the side arms (temples), which end up over the eyes in the AR try-on. Both the desktop app and the web page load glasses from **`assets/glasses/processed/`**, so a cleaned file there is used by both.

1. Put the product photo in `assets/glasses/` and run `python tools/prepare_glasses.py`. This removes the background and writes `assets/glasses/processed/<name>.png`.
2. Erase the arms: `python tools/clean_glasses.py <name>` (or with no name, to go through every image).
   - Left-click and drag erases. `[` / `]` makes the brush smaller or bigger. `Z` undoes a stroke, `R` resets the image.
   - `S` saves and moves to the next image. `N` skips it. `Q` quits.
3. Saving keeps the image the same size, so the glasses stay the same width and position on the face. It also softens the cut edges slightly.
   - The first save backs up the untouched file to `assets/glasses/originals/`. That backup is never overwritten; copy it back into `processed/` to start again.
   - `prepare_glasses.py` skips any image that has a backup in `originals/`, so re-running it won't undo your cleaning.
4. Optional, for the web version: add a line for the new image to `assets/glasses/glasses_styles.json` (for example `"glasses14": {"styles": ["round"]}`), so it appears under matching recommendations. Without it, the image still shows in the Previous / Next list.

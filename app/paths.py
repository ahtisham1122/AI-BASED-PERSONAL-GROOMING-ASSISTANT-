"""
Every folder the app reads from or writes to, built from this file's own
location -- so the app works no matter which folder it's started from.
"""
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
MODELS_DIR  = ROOT / 'models'
ASSETS_DIR  = ROOT / 'assets'
DATA_DIR    = ROOT / 'data'     # runtime output: app.log, feedback.json, snapshots/, debug/
DATASET_DIR = ROOT.parent / 'face_shape_dataset'   # optional, lives outside the project

DATA_DIR.mkdir(exist_ok=True)

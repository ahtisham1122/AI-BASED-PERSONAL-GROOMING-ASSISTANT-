"""
Measures the face shape CNN's real accuracy on the held-out TEST photos
(../face_shape_dataset/testing_set), using exactly the app's own crop
(BlazeFace -> square box + 25% margin -> black padding -> 224x224) and
the app's own loaded TFLite interpreter. Also runs the Keras model if
present. Changes nothing in the app.

    python tools/evaluate_tflite.py            # writes reports/tflite_evaluation.txt
"""
import glob
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from app import face_shape_model as fsm
from app.paths import DATASET_DIR, ROOT

TEST_DIR = DATASET_DIR / 'testing_set'
REPORT_PATH = ROOT / 'reports' / 'tflite_evaluation.txt'


def preprocess(path):
    """Same steps as fsm.predict_shape_probs: app crop, BGR->RGB, raw 0-255 float32 (the model rescales itself)."""
    img = cv2.imread(path)
    if img is None:
        return None
    crop = fsm._crop_and_resize(img, img.shape[1], img.shape[0])
    if crop is None:
        return None
    return np.expand_dims(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32), axis=0)


def predict_tflite(batch):
    fsm._interpreter.set_tensor(fsm._input_index, batch)
    fsm._interpreter.invoke()
    return fsm._interpreter.get_tensor(fsm._output_index)[0]


def load_keras():
    if not os.path.exists(fsm.KERAS_PATH):
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(fsm.KERAS_PATH)


def timed(fn, batch):
    t0 = time.perf_counter()
    probs = fn(batch)
    return int(np.argmax(probs)), (time.perf_counter() - t0) * 1000


def metrics_report(name, y_true, y_pred, times_ms, classes):
    n = len(classes)
    cm = np.zeros((n, n), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = np.trace(cm) / cm.sum()
    lines = [f"== {name} ==",
             f"Accuracy: {acc * 100:.2f}%  ({np.trace(cm)}/{cm.sum()})",
             f"Average prediction time: {np.mean(times_ms):.2f} ms (model call only, crop excluded)",
             "",
             f"{'Class':<8}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}"]
    f1s = []
    for i, c in enumerate(classes):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        lines.append(f"{c:<8}{prec:>10.3f}{rec:>10.3f}{f1:>10.3f}{cm[i, :].sum():>10}")
    lines.append(f"{'Macro F1':<8}{'':>20}{np.mean(f1s):>10.3f}")
    lines += ["", "Confusion matrix (rows = true, columns = predicted):",
              " " * 9 + "".join(f"{c:>8}" for c in classes)]
    lines += [f"{c:<9}" + "".join(f"{v:>8}" for v in cm[i]) for i, c in enumerate(classes)]
    return lines


def main():
    classes = fsm.CLASS_NAMES
    if fsm.MODE != 'tflite':
        sys.exit(f"TFLite model didn't load (mode = {fsm.MODE}); nothing to evaluate.")
    keras_model = load_keras()
    predict_keras = (lambda b: keras_model.predict(b, verbose=0)[0]) if keras_model else None  # same call the app uses

    y_true, tfl_pred, tfl_ms, ker_pred, ker_ms = [], [], [], [], []
    no_face, unreadable, total = {c: 0 for c in classes}, 0, 0
    for label, cls in enumerate(classes):
        paths = sorted(glob.glob(str(TEST_DIR / cls / '*')))
        print(f"{cls}: {len(paths)} images")
        for path in paths:
            total += 1
            batch = preprocess(path)
            if batch is None:
                if cv2.imread(path) is None:
                    unreadable += 1
                else:
                    no_face[cls] += 1
                continue
            if not y_true:  # one untimed warm-up call each: the first call includes one-off setup cost
                predict_tflite(batch)
                if predict_keras:
                    predict_keras(batch)
            y_true.append(label)
            p, ms = timed(predict_tflite, batch); tfl_pred.append(p); tfl_ms.append(ms)
            if predict_keras:
                p, ms = timed(predict_keras, batch); ker_pred.append(p); ker_ms.append(ms)

    lines = [f"Face shape CNN evaluation -- {datetime.now():%Y-%m-%d %H:%M}",
             f"Test folder: {TEST_DIR}",
             f"TFLite: {fsm.TFLITE_PATH}",
             f"Keras:  {fsm.KERAS_PATH if keras_model else 'not found, skipped'}",
             "Preprocessing: app's own _crop_and_resize (BlazeFace, square box + 25% margin, black padding, 224x224), BGR->RGB, raw 0-255 float32",
             "",
             f"Images found: {total}   evaluated: {len(y_true)}   skipped (no face detected): {sum(no_face.values())} {no_face}   unreadable: {unreadable}",
             ""]
    lines += metrics_report("TFLite", y_true, tfl_pred, tfl_ms, classes) + [""]
    if predict_keras:
        lines += metrics_report("Keras", y_true, ker_pred, ker_ms, classes) + [""]
        agree = np.mean(np.array(tfl_pred) == np.array(ker_pred)) * 100
        lines.append(f"TFLite and Keras agree on {agree:.2f}% of images")

    report = "\n".join(lines)
    print("\n" + report)
    os.makedirs(REPORT_PATH.parent, exist_ok=True)
    REPORT_PATH.write_text(report + "\n", encoding='utf-8')
    print(f"\nSaved {REPORT_PATH}")


if __name__ == '__main__':
    main()

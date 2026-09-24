"""
One-off diagnostic: takes 4 random images per class (20 total) from
face_shape_dataset/testing_set, crops them with the exact same crop
function the live app now uses (face_shape_model._crop_and_resize,
driven by the BlazeFace detector + 25% margin + padded-square, matching
how the training crops were built), runs both the .keras and .tflite
models on the identical crop, and prints predicted vs true label +
confidence for each.
"""
import cv2
import numpy as np
import random
import glob
import tensorflow as tf

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # project root
from app import face_shape_model as fsm
from app.paths import DATASET_DIR

random.seed(42)

# Load both models directly here, independent of which one face_shape_model
# itself picked (MODE only loads keras if tflite failed to load).
keras_model = tf.keras.models.load_model(fsm.KERAS_PATH)
interpreter  = tf.lite.Interpreter(model_path=fsm.TFLITE_PATH)
interpreter.allocate_tensors()
in_idx  = interpreter.get_input_details()[0]['index']
out_idx = interpreter.get_output_details()[0]['index']

SHAPES = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
samples = []
for shape in SHAPES:
    paths = glob.glob(str(DATASET_DIR / 'testing_set' / shape / '*'))
    samples.extend((p, shape) for p in random.sample(paths, min(4, len(paths))))

keras_correct  = 0
tflite_correct = 0
total = 0

for path, true_shape in samples:
    img = cv2.imread(path)
    if img is None:
        print(f"{path}: couldn't read image")
        continue
    h, w = img.shape[:2]

    resized = fsm._crop_and_resize(img, w, h)
    if resized is None:
        print(f"{path} (true={true_shape}): no face detected by BlazeFace, skipped")
        continue
    rgb   = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    batch = np.expand_dims(rgb.astype(np.float32), axis=0)

    k_probs = keras_model.predict(batch, verbose=0)[0]
    k_idx   = int(np.argmax(k_probs))
    k_label, k_conf = fsm.CLASS_NAMES[k_idx], float(k_probs[k_idx]) * 100

    interpreter.set_tensor(in_idx, batch)
    interpreter.invoke()
    t_probs = interpreter.get_tensor(out_idx)[0]
    t_idx   = int(np.argmax(t_probs))
    t_label, t_conf = fsm.CLASS_NAMES[t_idx], float(t_probs[t_idx]) * 100

    total += 1
    keras_correct  += (k_label == true_shape)
    tflite_correct += (t_label == true_shape)

    print(f"{path} true={true_shape:7s} | "
          f"keras={k_label:7s} ({k_conf:4.1f}%) | "
          f"tflite={t_label:7s} ({t_conf:4.1f}%)")

print(f"\nkeras accuracy:  {keras_correct}/{total} ({100*keras_correct/total:.0f}%)")
print(f"tflite accuracy: {tflite_correct}/{total} ({100*tflite_correct/total:.0f}%)")

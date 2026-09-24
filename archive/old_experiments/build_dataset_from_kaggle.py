"""
ARCHIVED (superseded by the MobileNetV2 CNN trained via train_face_shape_cnn_v2.ipynb): this script built the ratio-based CSV for the old Random Forest approach, which the CNN replaced for materially better accuracy.

Builds a training CSV from the Kaggle face-shape dataset, using the exact
same MediaPipe landmarker setup and ratio math (compute_face_ratios) as
the live camera app, so the training data matches what it measures.
"""
import cv2
import mediapipe as mp
import os
import csv
import glob
from collections import Counter

from face_tracking import MODEL_PATH, download_model
from face_shape_model import compute_face_ratios

DATASET_DIR  = 'face_shape_dataset'
SPLITS       = ['training_set', 'testing_set']
SHAPES       = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
OUTPUT_CSV   = 'face_shape_training_data.csv'
RATIO_FIELDS = ['hr', 'fr', 'jr', 'flr', 'jaw_angle', 'temple_ratio', 'chin_ratio']


def main():
    download_model()

    BaseOptions           = mp.tasks.BaseOptions
    FaceLandmarker        = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
    )

    image_paths = []
    for split in SPLITS:
        for shape in SHAPES:
            folder = os.path.join(DATASET_DIR, split, shape)
            for path in glob.glob(os.path.join(folder, '*')):
                image_paths.append((path, shape))

    total = len(image_paths)
    print(f"Found {total} images.")

    processed = Counter()
    skipped   = Counter()

    with FaceLandmarker.create_from_options(options) as landmarker, \
            open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(RATIO_FIELDS + ['label'])

        for i, (path, shape) in enumerate(image_paths, 1):
            img = cv2.imread(path)
            ratios = None

            if img is not None:
                h, w = img.shape[:2]
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                )
                result = landmarker.detect(mp_image)
                if result.face_landmarks:
                    ratios = compute_face_ratios(result.face_landmarks[0], w, h)

            if ratios is None:
                skipped[shape] += 1
            else:
                writer.writerow([ratios[field] for field in RATIO_FIELDS] + [shape])
                processed[shape] += 1

            if i % 10 == 0 or i == total:
                print(f"processed {i}/{total}")

    print("\n=== Summary ===")
    for shape in SHAPES:
        print(f"{shape}: {processed[shape]} usable, "
              f"{skipped[shape]} skipped (no face found)")
    print(f"\nTotal usable rows: {sum(processed.values())}")
    print(f"Total skipped: {sum(skipped.values())}")
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == '__main__':
    main()

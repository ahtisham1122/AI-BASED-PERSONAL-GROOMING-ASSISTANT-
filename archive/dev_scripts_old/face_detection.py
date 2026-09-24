# Old test version, superseded by main.py.
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# New MediaPipe API
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Download the model first
import urllib.request
import os

model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading face landmark model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        model_path
    )
    print("Model downloaded!")

# Setup options
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

# Open camera
cap = cv2.VideoCapture(0)
print("Camera started! Press Q to quit.")

with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Cannot access camera.")
            break

        # Flip frame like a mirror
        frame = cv2.flip(frame, 1)

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect landmarks
        result = landmarker.detect(mp_image)

        # Draw landmarks if face detected
        if result.face_landmarks:
            h, w, _ = frame.shape
            for face in result.face_landmarks:
                for landmark in face:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            cv2.putText(frame, "Face Detected!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Face Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("AI Grooming Assistant - Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Camera closed.")
# Old test version, superseded by main.py.
import cv2

print("Checking all cameras...")
print("=" * 40)

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Camera [{i}] — WORKING")
            print(f"  Resolution: "
                  f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}"
                  f"x"
                  f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
            )
        else:
            print(f"Camera [{i}] — Opens but no frame")
        cap.release()
    else:
        print(f"Camera [{i}] — Not found")

print("=" * 40)
print("Use the index that shows DroidCam resolution")
# Old test version, superseded by main.py.
import cv2

print("Showing Camera 0 - press any key to switch")

# Show camera 0
cap0 = cv2.VideoCapture(0)
while True:
    ret, frame = cap0.read()
    if ret:
        cv2.putText(frame, "Camera 0 - Laptop?",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF != 255:
        break
cap0.release()
cv2.destroyAllWindows()

print("Now showing Camera 1 - press any key to close")

# Show camera 1
cap1 = cv2.VideoCapture(1)
while True:
    ret, frame = cap1.read()
    if ret:
        cv2.putText(frame, "Camera 1 - DroidCam?",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF != 255:
        break
cap1.release()
cv2.destroyAllWindows()

print("Done! Now you know which index is DroidCam")
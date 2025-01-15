import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera initialization failed")
else:
    print("Camera is working")
    cap.release()

try:
    import cv2
    print("cv2 is accessible")
except ImportError:
    print("cv2 is not accessible")

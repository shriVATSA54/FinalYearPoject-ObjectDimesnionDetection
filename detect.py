import cv2

print("🔍 Searching for available cameras...\n")
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Camera found at index {i}")
        cap.release()

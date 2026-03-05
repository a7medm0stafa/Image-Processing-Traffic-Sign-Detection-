import cv2

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

if ret:
    cv2.imwrite("photo.jpg", frame)
    print("Photo saved as photo.jpg")
else:
    print("Failed to capture image")

cap.release()

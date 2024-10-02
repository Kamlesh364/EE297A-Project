from ultralytics import YOLO
import cv2

# Load model
model = YOLO('yolov8n.pt')

# define source
source_cap = cv2.VideoCapture('traffic.mp4')

while True:
    ret, frame = source_cap.read()
    # cv2.imshow('frame', frame)
    # Inference
    results = model(source=frame, show=True, save=False)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

source_cap.release()
cv2.destroyAllWindows()

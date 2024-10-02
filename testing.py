from ultralytics import YOLO
import cv2

# Load model
model = YOLO('yolov8n.pt')

# define source
source_cap = cv2.VideoCapture("assets/traffic.mp4")

while True:
    ret, frame = source_cap.read()
    results = model(source=frame, show=False, save=False)
    bbox = results.pandas().xyxy[0]
    cls = results.pandas().xyxy[0].name
    print(cls)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

source_cap.release()
cv2.destroyAllWindows()

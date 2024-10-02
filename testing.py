#!/usr/bin/python3

from ultralytics import YOLO
import cv2
from torch.cuda import is_available
from numpy.random import uniform

device = 'cuda' if is_available() else 'cpu'
print(f'Using {device} device')

# Load model
model = YOLO('yolov8n.pt')

# define source
source_cap = cv2.VideoCapture("assets/traffic.mp4")
COLORS = uniform(0, 255, size=(80, 3)) # random colors for different classes

while True:
    # Capture frame-by-frame
    ret, frame = source_cap.read()

    # Check if frame is None
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Copy frame for detection
    det = frame.copy()

    # Inference
    results = model.predict(frame, show=False, verbose=False, save=False, device=device)

    # Check if robot is detected
    if results[0].boxes.cpu().numpy().xyxy.shape[0] != 0:
        # Show results on image
        boxes = results[0].boxes.cpu().numpy().xyxy.astype(int)
        labels = results[0].boxes.cpu().numpy().cls
        conf = results[0].boxes.cpu().numpy().conf
        for box, label, conf in zip(boxes, labels, conf):
            x1, y1, x2, y2 = box
            cv2.rectangle(det, (x1, y1), (x2, y2), COLORS[int(label)], 1)
            cv2.putText(
                det,
                model.names[int(label)],
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                COLORS[int(label)],
                1,
            )

    # Display the resulting frame
    cv2.imshow("frame", cv2.hconcat([frame, det]))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
source_cap.release()
cv2.destroyAllWindows()

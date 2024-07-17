from ultralytics import YOLO
import cv2

# Load a pretrained YOLOv8 model
Model_Path = r"C:\Users\garne\Downloads\best.pt"
model = YOLO(Model_Path)
# "yolov5nu.pt"

# Specify WebCam path
cap = cv2.VideoCapture(0)

# f-b-f analysis
while True:
    connected, frame = cap.read()
    if connected:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow('Machine Vision', annotated_frame)

        boxes = None
        for result in results:
            boxes = result.boxes.xyxy

        if boxes is not None and boxes.size(0) > 0:
            print("Objects detected!")
            print(boxes)
        else:
            print("No objects detected.")

        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

# Release the webcam and close any open CV windows
cap.release()
cv2.destroyAllWindows()

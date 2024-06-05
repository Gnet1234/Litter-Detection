from ultralytics import YOLO
import cv2

# Load a pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

#Specify WebCam path
cap=cv2.VideoCapture(1)

#f-b-f analysis
while True:
    #Frame-by-frame video capture
    ret, frame= cap.read()

    #Object detection with YoloV8 model on the frame
    results= model(frame)

    #Image Segmentation
    for results in results:
        boxes = results.boxes
        for box in boxes:
            #Obtain coordinate
            x1,y1,x2,y2 = box.xyxy[0].tolist() #bounding box coordinate

            #obtain class label and confidence
            conf= box.conf[0]
            cls=box.cls[0]

            #draw boundary from obtained information
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Put the class name and confidence score
            cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Webcam Object Detection',frame)

# Quit if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break



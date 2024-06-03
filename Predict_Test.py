from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Run inference on an image
Image_Path = r'C:\Users\garne\Downloads\car3.jpg'
results = model(Image_Path)  # results list

# View results
for r in results:
    print(r.boxes)  # print the Boxes object containing the detection bounding boxes

    # Show image with bounding boxes
    r.show()
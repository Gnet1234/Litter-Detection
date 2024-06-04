from ultralytics import YOLO
import glob
import cv2

# loading a custom model
PATH = r'C:\Users\garne\runs\detect\train13\weights\best.pt'
model = YOLO(PATH)
Image_Path = r'C:\Users\garne\Downloads\car3.jpg'
results = model(Image_Path)  # results list

# View results
for r in results:
    print(r.boxes)  # print the Boxes object containing the detection bounding boxes

    # Show image with bounding boxes
    r.show()
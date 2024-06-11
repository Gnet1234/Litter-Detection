from ultralytics import YOLO
import torch

# Link to dataset: https://universe.roboflow.com/littereye/littereye

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(data=r'C:\Users\garne\anaconda3\envs\TensorFlowObjectDetection\TensorFlow\Car Dataset Yolov5\data.yaml', epochs=1, optimizer = 'Adam', plots = True, patience = 2)

import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")


# Print the custom model class
print("Custom model class:", model.model)

# Save the trained model's state dictionary
save_path = r'C:\Users\garne\anaconda3\envs\TensorFlowObjectDetection\TensorFlow\models\yolov5_car.pt'
torch.save(model.state_dict(), save_path)


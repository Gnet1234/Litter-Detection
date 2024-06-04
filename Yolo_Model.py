from ultralytics import YOLO
import torch

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(data=r'C:\Users\garne\anaconda3\envs\TensorFlowObjectDetection\TensorFlow\Car Dataset Yolov5\data.yaml', epochs=1)

# Print the custom model class
print("Custom model class:", model.model)

# Save the trained model's state dictionary
save_path = r'C:\Users\garne\anaconda3\envs\TensorFlowObjectDetection\TensorFlow\models\yolov5_car.pt'
torch.save(model.state_dict(), save_path)


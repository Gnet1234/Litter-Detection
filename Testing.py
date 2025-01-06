from ultralytics import YOLO
import cv2
import numpy as np

# Load a pretrained YOLOv8 model
Model_Path = r"C:\Users\garne\Downloads\best (1).pt"
model = YOLO('yolov8n-seg.pt')

# Scaling Ratio for pixels in cm/pixel
PSF = 1.2

# Scaling Ratio for distance (This would be found by tracking the distance of the drone from the ground)
DSF = 0.2

# Specify WebCam path
cap = cv2.VideoCapture(0)

# f-b-f analysis
while True:
    # Frame-by-frame video capture
    connected, frame = cap.read()
    if connected:
        # Obtain results from detection with YOLOv8 model on the frame
        results = model(frame)

        # Plot boundary & image masking from obtained results in f2f format
        annotated_frame = results[0].plot()
        # Calculate area of detected objects and their heights
        for i, (mask, bbox) in enumerate(zip(results[0].masks.data, results[0].boxes.xyxy)):
            # Convert mask to binary
            binary_mask = mask.cpu().numpy() > 0  # Assuming masks are in a PyTorch tensor
            object_area = np.sum(binary_mask)  # Count non-zero pixels

            # Calculate pixel height of the object from bounding box
            x_min, y_min, x_max, y_max = map(int, bbox)
            pixel_height = y_max - y_min

            print(f"Object {i + 1}:")
            print(f" - Area: {object_area} pixels")
            print(f" - Pixel Height: {pixel_height} pixels")

            # Convert the pixels to cm with the use of the ratio
            NH = pixel_height * PSF

            # Calculate the true height by multiplying the converted pixel height to cm and multiplying by the distance ratio
            True_Height = NH * DSF

            print(f"The height of the object in cm is: {True_Height}")

            # Optionally draw the bounding box and display the pixel height on the frame
            #cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            #cv2.putText(annotated_frame, f"Height: {pixel_height}px", (x_min, y_min - 10),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display results
        cv2.imshow('Machine Vision', annotated_frame)

        # Stop Process press 'q'
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()

from ultralytics import YOLO
import cv2
import threading
import time

# Finds the model which is saved as a .pt file, and saves it to a variable.
Model_Path = r"/home/labpc/Downloads/yolov5n.pt"
model = YOLO(Model_Path)

# Specify WebCam path
cap = cv2.VideoCapture(0)


class LogicTest:
    def __init__(self):
        self.running = threading.Event()  # Use threading.Event for controlling the loop
        self.running.set()  # Start the event as set
        self.results = []  # Shared list for results

    def frame_data(self):
        while self.running.is_set():
            connected, frame = cap.read()
            if connected:
                results = model(frame)
                self.results = results[0].boxes.xyxy  # Store the detection results

                annotated_frame = results[0].plot()  # Annotate the frame with detections
                cv2.imshow('Machine Vision', annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running.clear()  # Stop the loop when 'q' is pressed

    def control(self):
        while self.running.is_set():
            time.sleep(0.1)  # Sleep briefly to avoid busy waiting
            if self.results is not None and len(self.results) > 0:
                print("Objects detected!")

            else:
                print("No objects detected.")


def test():
    logictest = LogicTest()

    # Start the frame processing in a separate thread
    frame_thread = threading.Thread(target=logictest.frame_data)
    frame_thread.start()

    # Start the control logic in the main thread
    logictest.control()


if __name__ == "__main__":
    test()
    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

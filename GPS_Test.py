from __future__ import print_function
import csv
import cv2
import os
import queue
import shlex
import subprocess
import tempfile
import threading
import multiprocessing
import inputs  # Importing the gamepad library
import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, moveBy
from olympe.messages.ardrone3.PilotingState import GpsLocationChanged, FlyingStateChanged
from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged, HomeChanged
from ultralytics import YOLO
import time
import pandas as pd
from datetime import datetime


olympe.log.update_config({"loggers": {"olympe": {"level": "WARNING"}}})

# Gets the drone IP address, and the port so the code can connect to it.
DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")
DRONE_RTSP_PORT = os.environ.get("DRONE_RTSP_PORT")

# Finds the model which is saved as a .pt file, and saves it to a variable.
Model_Path = r"/home/labpc/Downloads/yolov5n.pt"
model = YOLO(Model_Path)

# Define debounce delay in seconds (adjust as needed)
DEBOUNCE_DELAY = 0.3

# Define a global DataFrame to store GPS data
gps_data_df = pd.DataFrame(columns=['Timestamp', 'Latitude', 'Longitude', 'Altitude'])



class StreamingExample:
    def __init__(self):
        # Create the olympe.Drone object from its IP address.
        self.drone = olympe.Drone(DRONE_IP)
        # Creates a temporary for storing output files.
        self.tempd = tempfile.mkdtemp(prefix="olympe_streaming_test_")
        # Prints the path for the temporary directory.
        print(f"Olympe streaming example output dir: {self.tempd}")
        """""""""
        # Opens an empty list to store h264 statistics.
        self.h264_frame_stats = []
        # Opens a temporary directory for writing video stream statistics.
        self.h264_stats_file = open(os.path.join(self.tempd, "h264_stats.csv"), "w+")
        self.h264_stats_writer = csv.DictWriter(
            self.h264_stats_file, ["fps", "bitrate"]
        )
        self.h264_stats_writer.writeheader()
        self.frame_queue = multiprocessing.Queue()
        self.processed_frame_queue = multiprocessing.Queue()
        self.processes = []
        self.running = multiprocessing.Event()
        self.running.set()
        """""
    def start(self):
        # Connect to drone
        assert self.drone.connect(retry=3)
        """""""""
        if DRONE_RTSP_PORT is not None:
            self.drone.streaming.server_addr = f"{DRONE_IP}:{DRONE_RTSP_PORT}"
        
        # You can record the video stream from the drone if you plan to do some
        # post processing.
        self.drone.streaming.set_output_files(
            video=os.path.join(self.tempd, "streaming.mp4"),
            metadata=os.path.join(self.tempd, "streaming_metadata.json"),
        )

        # Setup your callback functions to do some live video processing
        self.drone.streaming.set_callbacks(
            raw_cb=self.yuv_frame_cb,
            h264_cb=self.h264_frame_cb,
            start_cb=self.start_cb,
            end_cb=self.end_cb,
            flush_raw_cb=self.flush_cb,
        )
        # Start video streaming
        self.drone.streaming.start()

        # Start multiple processing threads
        num_processes = multiprocessing.cpu_count()
        self.processes = [multiprocessing.Process(target=self.yuv_frame_processing, args=(self.running,))
                          for _ in range(num_processes)]
        for p in self.processes:
            p.start()

        # Start the video output processing thread
        self.output_thread = threading.Thread(target=self.process_video_output)
        self.output_thread.start()
        """""
    def stop(self):
        """""""""
        self.running.clear()
        for p in self.processes:
            p.join()

        self.output_thread.join()

        # Properly stop the video stream and disconnect
        assert self.drone.streaming.stop()
        assert self.drone.disconnect()
        self.h264_stats_file.close()
        """""
    """""""""
    def yuv_frame_cb(self, yuv_frame):
       
        yuv_frame.ref()
        info = yuv_frame.info()
        height, width = info["raw"]["frame"]["info"]["height"], info["raw"]["frame"]["info"]["width"]
        yuv_data = yuv_frame.as_ndarray()
        self.frame_queue.put((yuv_data, height, width))
        yuv_frame.unref()

    def yuv_frame_processing(self, running):
        while running.is_set():
            try:
                yuv_data, height, width = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            # Convert YUV data to OpenCV format
            yuv_data = yuv_data.reshape((height * 3 // 2, width))
            bgr_frame = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2BGR_I420)
            results = model(bgr_frame)
            # Plot boundary & image masking from obtained results in f2f format
            annotated_frame = results[0].plot()
            self.processed_frame_queue.put(annotated_frame)

    def process_video_output(self):
        while self.running.is_set():
            try:
                frame = self.processed_frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            cv2.imshow('Machine Vision', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running.clear()
                break

        cv2.destroyAllWindows()

    def flush_cb(self, stream):
        if stream["vdef_format"] != olympe.VDEF_I420:
            return True
        while not self.frame_queue.empty():
            self.frame_queue.get_nowait()
        return True

    def start_cb(self):
        pass

    def end_cb(self):
        pass

    def h264_frame_cb(self, h264_frame):
     

        # Get a ctypes pointer and size for this h264 frame
        frame_pointer, frame_size = h264_frame.as_ctypes_pointer()

        # For this example we will just compute some basic video stream stats
        # (bitrate and FPS) but we could choose to resend it over another.
        # interface or to decode it with our preferred hardware decoder.

        # Compute some stats and dump them in a csv file
        info = h264_frame.info()
        frame_ts = info["ntp_raw_timestamp"]
        if not bool(info["is_sync"]):
            while len(self.h264_frame_stats) > 0:
                start_ts, _ = self.h264_frame_stats[0]
                if (start_ts + 1e6) < frame_ts:
                    self.h264_frame_stats.pop(0)
                else:
                    break
            self.h264_frame_stats.append((frame_ts, frame_size))
            h264_fps = len(self.h264_frame_stats)
            h264_bitrate = 8 * sum(map(lambda t: t[1], self.h264_frame_stats))
            self.h264_stats_writer.writerow({"fps": h264_fps, "bitrate": h264_bitrate})
        """""

    def fly(self):
        global gps_data_df  # Declare gps_data_df as global

        # Initialize debounce timers for each button
        debounce_tl = time.time()
        debounce_tr = time.time()
        debounce_tul = time.time()
        debounce_tur = time.time()
        debounce_au = time.time()
        debounce_ad = time.time()

        # Checks if motion occurred, and records the location data if it did.
        Motion_Check = False

        try:
            # Prints the location of the drone before taking off.
            # Send the SetHome command


            print("GPS position before take-off :", self.drone.get_state(HomeChanged))
            while True:

                def main(stdscr):
    # Initialize curses
    curses.cbreak()
    stdscr.keypad(True)
    stdscr.nodelay(True)  # Make getch() non-blocking

    stdscr.addstr(0, 0, "Press 'r' to record GPS data. Press 'q' to quit.")

    while True:
        key = stdscr.getch()
        if key == ord('r'):  # Replace 'r' with the key you want to check
            # Wait for GPS location change
            gps_data = self.drone.get_state(GpsLocationChanged)

            # Extract coordinates, and record the time when it happens.
            timestamp = datetime.now()
            latitude = gps_data['latitude']
            longitude = gps_data['longitude']
            altitude = gps_data['altitude']

            # Print the GPS coordinates
            stdscr.addstr(1, 0, f"Latitude: {latitude:.7f}, Longitude: {longitude:.7f}, Altitude: {altitude:.2f}")
            stdscr.refresh()

            # Add the recorded data to the data frame (gps_data_df).
            gps_data_df = pd.concat([gps_data_df, pd.DataFrame({
                'Timestamp': [timestamp],
                'Latitude': [latitude],
                'Longitude': [longitude],
                'Altitude': [altitude]
            })], ignore_index=True)

            # Wait for the key to be released to avoid multiple recordings
            while stdscr.getch() == ord('r'):
                pass

        elif key == ord('q'):  # Press 'q' to exit the loop
            break

if __name__ == "__main__":
    curses.wrapper(main)

                # Creates an event based on the input of the gamepad.
                events = inputs.get_gamepad()

                # For loop that monitors the game-pads input, and produces an output that moves the drone.
                for event in events:
                    # There are two different event types for the gamepad, absolute which have a range of -35000 to 35000, and key which have are either 1 or 0.
                    # The key events are the buttons, while absolute events are the sticks and triggers.
                    if event.ev_type == 'Absolute':
                        # event.code is used to recognize the specific buttons.
                        if event.code == 'ABS_RY':
                            # This variable stores the value produced from pressing the button.
                            y_value2 = event.state
                            # Debounce time is used to limit the amount commands that are registered, since the buttons can be sensetive.
                            if y_value2 > 30000 and (time.time() - debounce_ad) > DEBOUNCE_DELAY:  # Downward
                                # This moves the drone.
                                self.drone(moveBy(0, 0, -1, 0)).wait()
                                print("Moving Downwards")
                                # This resets the debounce_ad, so the button can be registered again.
                                debounce_ad = time.time()
                                # Gives the command to print, and store the location data.
                                Motion_Check = True
                            elif y_value2 < -30000 and (time.time() - debounce_au) > DEBOUNCE_DELAY:  # Upwards
                                self.drone(moveBy(0, 0, 1, 0)).wait()
                                print("Moving Upwards")
                                debounce_au = time.time()
                                Motion_Check = True
                            elif abs(y_value2) <= 30000:  # Deadzone for y_value2
                                self.drone(moveBy(0, 0, 0, 0)).wait()  # Stop movement
                        elif event.code == 'ABS_Z':
                            if event.state <= 1000 and (time.time() - debounce_tul) > DEBOUNCE_DELAY:
                                self.drone(moveBy(0, -1, 0, 0)).wait()
                                print("Moving Left")
                                debounce_tul = time.time()
                                Motion_Check = True
                        elif event.code == 'ABS_RZ':
                            if event.state <= 1000 and (time.time() - debounce_tur) > DEBOUNCE_DELAY:
                                self.drone(moveBy(0, 1, 0, 0)).wait()
                                print("Moving Right")
                                debounce_tur = time.time()
                                Motion_Check = True
                    elif event.ev_type == 'Key':
                        if event.code == 'BTN_SOUTH' and event.state == 1:  # A button
                            self.drone(TakeOff())
                            print("TakeOff")
                            Motion_Check = True
                        elif event.code == 'BTN_EAST' and event.state == 1:  # B button
                            self.drone(Landing())
                            print("Landing")
                            Motion_Check = True
                        elif event.code == 'BTN_WEST' and event.state == 1:  # Y button
                            self.drone(moveBy(0, 0, 0, 1)).wait()
                            print("Turning clockwise")
                            Motion_Check = True
                        elif event.code == 'BTN_NORTH' and event.state == 1:  # X button
                            self.drone(moveBy(0, 0, 0, -1)).wait()
                            print("Turning counter clockwise")
                            Motion_Check = True
                        elif event.code == 'BTN_TL':
                            if event.state == 1 and (time.time() - debounce_tl) > DEBOUNCE_DELAY:
                                self.drone(moveBy(-1, 0, 0, 0)).wait()
                                print("Moving Backwards")
                                debounce_tl = time.time()
                                Motion_Check = True
                        elif event.code == 'BTN_TR':
                            if event.state == 1 and (time.time() - debounce_tr) > DEBOUNCE_DELAY:
                                self.drone(moveBy(1, 0, 0, 0)).wait()
                                print("Moving Forwards")
                                debounce_tr = time.time()
                                Motion_Check = True

                        # Add more mappings for other buttons as needed
        except KeyboardInterrupt:
            print("Stopping control with Xbox controller...")

            print("GPS monitoring stopped.")
            # Save data to Excel file
            save_to_excel(gps_data_df)

        print("Landing...")
        self.drone(Landing() >> FlyingStateChanged(state="landed", _timeout=5)).wait()
        print("Landed\n")



    def replay_with_vlc(self):
        # Replay this MP4 video file using VLC
        mp4_filepath = os.path.join(self.tempd, "streaming.mp4")
        subprocess.run(shlex.split(f"vlc --play-and-exit {mp4_filepath}"), check=True)


def save_to_excel(df):
    # Save DataFrame to Excel file
    filename = 'gps_data4.xlsx'
    df.to_excel(filename, index=False, engine='openpyxl')
    print(f"GPS data saved to {filename}")



def test_streaming():
    streaming_example = StreamingExample()
    # Start the video stream
    streaming_example.start()
    # Perform some live video processing while the drone is flying
    streaming_example.fly()
    # Stop the video stream
    #streaming_example.stop()
    # Recorded video stream postprocessing
    # streaming_example.replay_with_vlc()


if __name__ == "__main__":
    test_streaming()

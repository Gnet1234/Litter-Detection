from __future__ import print_function
import csv
import cv2
import os
import queue
import shlex
import subprocess
import tempfile
import threading
import inputs  # Importing the gamepad library
import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, Landing
from olympe.messages.ardrone3.Piloting import moveBy
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.video.renderer import PdrawRenderer
from ultralytics import YOLO


olympe.log.update_config({"loggers": {"olympe": {"level": "WARNING"}}})

DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")
DRONE_RTSP_PORT = os.environ.get("DRONE_RTSP_PORT")

Model_Path = r"/home/labpc/Downloads/yolov8n-seg.pt"
model = YOLO(Model_Path)


class StreamingExample:
    def __init__(self):
        # Create the olympe.Drone object from its IP address
        self.drone = olympe.Drone(DRONE_IP)
        self.tempd = tempfile.mkdtemp(prefix="olympe_streaming_test_")
        print(f"Olympe streaming example output dir: {self.tempd}")
        self.h264_frame_stats = []
        self.h264_stats_file = open(os.path.join(self.tempd, "h264_stats.csv"), "w+")
        self.h264_stats_writer = csv.DictWriter(
            self.h264_stats_file, ["fps", "bitrate"]
        )
        self.h264_stats_writer.writeheader()
        self.frame_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.yuv_frame_processing)
        self.renderer = None

    def start(self):
        # Connect to drone
        assert self.drone.connect(retry=3)

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
        #self.renderer = PdrawRenderer(pdraw=self.drone.streaming)  #OLYMPE VIDEO
        self.running = True
        self.processing_thread.start()

    def stop(self):
        self.running = False
        self.processing_thread.join()
        if self.renderer is not None:
            self.renderer.stop()
        # Properly stop the video stream and disconnect
        assert self.drone.streaming.stop()
        assert self.drone.disconnect()
        self.h264_stats_file.close()

    def yuv_frame_cb(self, yuv_frame):
        """
        This function will be called by Olympe for each decoded YUV frame.

            :type yuv_frame: olympe.VideoFrame
        """
        yuv_frame.ref()
        self.frame_queue.put_nowait(yuv_frame)

    def yuv_frame_processing(self):
        while self.running:
            try:
                yuv_frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            # Convert YUV frame to OpenCV format
            info = yuv_frame.info()
            height, width = info["raw"]["frame"]["info"]["height"], info["raw"]["frame"]["info"]["width"]
            yuv_data = yuv_frame.as_ndarray()
            yuv_data = yuv_data.reshape((height * 3 // 2, width))
            bgr_frame = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2BGR_I420)
            results = model(bgr_frame)
            # Plot boundary & image masking from obtained results in f2f format
            annotated_frame = results[0].plot()
            # Display the frame
            cv2.imshow('Machine Vision', annotated_frame)
            # cv2.imshow("Drone Feed", bgr_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            yuv_frame.unref()
        cv2.destroyAllWindows()

    def flush_cb(self, stream):
        if stream["vdef_format"] != olympe.VDEF_I420:
            return True
        while not self.frame_queue.empty():
            self.frame_queue.get_nowait().unref()
        return True

    def start_cb(self):
        pass

    def end_cb(self):
        pass

    def h264_frame_cb(self, h264_frame):
        """
        This function will be called by Olympe for each new h264 frame.

            :type yuv_frame: olympe.VideoFrame
        """

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

    def show_yuv_frame(self, window_name, yuv_frame):
        # the VideoFrame.info() dictionary contains some useful information
        # such as the video resolution
        info = yuv_frame.info()

        height, width = (  # noqa
            info["raw"]["frame"]["info"]["height"],
            info["raw"]["frame"]["info"]["width"],
        )

        # yuv_frame.vmeta() returns a dictionary that contains additional
        # metadata from the drone (GPS coordinates, battery percentage, ...)

        # convert pdraw YUV flag to OpenCV YUV flag
        # import cv2
        # cv2_cvt_color_flag = {
        #     olympe.VDEF_I420: cv2.COLOR_YUV2BGR_I420,
        #     olympe.VDEF_NV12: cv2.COLOR_YUV2BGR_NV12,
        # }[yuv_frame.format()]

    def fly(self):

        try:
            while True:
                events = inputs.get_gamepad()
                for event in events:
                    if event.ev_type == 'Absolute':
                        if event.code == 'ABS_X':
                            x_value = event.state
                            if x_value > 20000:  # Right
                                self.drone(moveBy(0, 1, 0, 0)).wait()
                                print("Moving Right")
                            elif x_value < -20000:  # Left
                                self.drone(moveBy(0, -1, 0, 0)).wait()
                                print("Moving Left")
                            elif abs(x_value) <= 20000:  # Deadzone for x_value
                                self.drone(moveBy(0, 0, 0, 0)).wait()  # Stop movement
                        elif event.code == 'ABS_Y':
                            y_value = event.state
                            if y_value > 20000:  # Backwards
                                self.drone(moveBy(-1, 0, 0, 0)).wait()
                                print("Moving Backwards")
                            elif y_value < -20000:  # Forwards
                                self.drone(moveBy(1, 0, 0, 0)).wait()
                                print("Moving Forwards")
                            elif abs(y_value) <= 20000:  # Deadzone for y_value
                                self.drone(moveBy(0, 0, 0, 0)).wait()  # Stop movement
                        elif event.code == 'ABS_RY':
                            y_value2 = event.state
                            if y_value2 > 20000:  # Downward
                                self.drone(moveBy(0, 0, -1, 0)).wait()
                                print("Moving Downwards")
                            elif y_value2 < -20000:  # Upwards
                                self.drone(moveBy(0, 0, 1, 0)).wait()
                                print("Moving Upwards")
                            elif abs(y_value2) <= 20000:  # Deadzone for y_value2
                                self.drone(moveBy(0, 0, 0, 0)).wait()  # Stop movement
                    elif event.ev_type == 'Key':
                        if event.code == 'BTN_SOUTH' and event.state == 1:  # A button
                            self.drone(TakeOff())
                            print("TakeOff")
                        elif event.code == 'BTN_EAST' and event.state == 1:  # B button
                            self.drone(Landing())
                            print("Landing")
                        elif event.code == 'BTN_WEST' and event.state == 1:  # Y button
                            self.drone(moveBy(0, 0, 0, 1)).wait()
                            print("Turning clockwise")
                        elif event.code == 'BTN_NORTH' and event.state == 1:  # X button
                            self.drone(moveBy(0, 0, 0, -1)).wait()
                            print("Turning counter clockwise")
                        # Add more mappings for other buttons as needed
        except KeyboardInterrupt:
            print("Stopping control with Xbox controller...")

        print("Landing...")
        self.drone(Landing() >> FlyingStateChanged(state="landed", _timeout=5)).wait()
        print("Landed\n")

    def replay_with_vlc(self):
        # Replay this MP4 video file using VLC
        mp4_filepath = os.path.join(self.tempd, "streaming.mp4")
        subprocess.run(shlex.split(f"vlc --play-and-exit {mp4_filepath}"), check=True)


def test_streaming():
    streaming_example = StreamingExample()
    # Start the video stream
    streaming_example.start()
    # Perform some live video processing while the drone is flying
    streaming_example.fly()
    # Stop the video stream
    streaming_example.stop()
    # Recorded video stream postprocessing
    # streaming_example.replay_with_vlc()


if __name__ == "__main__":
    test_streaming()

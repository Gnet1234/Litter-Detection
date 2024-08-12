import olympe
from olympe.messages.gimbal import set_target
import os
import time

# Connect to the drone
DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")
drone = olympe.Drone(DRONE_IP)

try:
    # Connect to the drone
    drone.connect()

    # Set the gimbal target orientation
    # Example: Set gimbal to 30 degrees pitch, 0 degrees roll, and 0 degrees yaw
    gimbal_command = set_target(
        gimbal_id=0,
        control_mode="position",  # Use "velocity" for smooth movement
        yaw_frame_of_reference="absolute",  # Can be "relative" or "absolute"
        yaw=0.0,
        pitch_frame_of_reference="absolute",  # Use "absolute" or "relative"
        pitch=0.0,
        roll_frame_of_reference="absolute",  # Use "absolute" or "relative"
        roll=0.0,
    )

    # Send the command and wait for the completion
    command_result = drone(gimbal_command).wait()

    # Check if the command was successful
    if command_result.success():
        print("Gimbal target orientation set successfully.")
    else:
        print("Failed to set gimbal target orientation.")

    # Keep the drone and gimbal stable for a moment
    time.sleep(5)

finally:
    # Disconnect from the drone
    drone.disconnect()

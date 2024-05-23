import olympe
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# IP address of the drone
DRONE_IP = "192.168.42.21"  # Replace with your drone's IP address

def test_connection():
    # Create the olympe.Drone object from its IP address
    drone = olympe.Drone(DRONE_IP)

    try:
        # Connect to the drone
        drone.connect()

        # If the connection is successful, print a success message
        print("Drone connection successful!")

        # Disconnect from the drone
        drone.disconnect()
    except Exception as e:
        # If an error occurs during connection, print the error message
        print(f"Error connecting to the drone: {e}")

if __name__ == "__main__":
    test_connection()

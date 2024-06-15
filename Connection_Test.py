import olympe
import os
import logging

DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Used to activate virtual environment: source /mnt/c/Drone/venv/bin/activate
# Used to Start code: python3 /mnt/c/Drone/venv/Connect.py 

# It can connect, but there is a syncronization issue causing it to disconnect. 

def test_physical_drone():
    drone = olympe.Drone(DRONE_IP)
    drone.connect()
    drone.disconnect()


if __name__ == "__main__":
    test_physical_drone()

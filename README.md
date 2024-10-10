The goal of this project is to use a Parrot Anafi drone, and object detection model to improve the efficiency of litter audits. Parrot Anafi was chosen because of its library which allows for communication between the drone, and other devices. This can be done through the Olympe library that is used for communication in the codes. Yolo is being used as the object detection model, for its ease of use and support. 

Parrot Olympe can be installed with this:
pip3 install parrot-olympe

More information about the library can be found here: https://developer.parrot.com/docs/olympe/index.html

Yolo can be installed with this:
pip install ultralytics

More information can be found here:
https://docs.ultralytics.com

The following codes show the steps that were taken to develop the objective, and are in the Important Files forlder:
- GPU-TRAIN: This code trains the model, using the GPU of the computer. It uses cuda to achieve this, and it was used with a windows laptop with a nvidia GPU. 

- Video_And_Motion: This code captures the video from the drone, and passes it through the object detection model. It also uses the inputs library which allows for the drone to be controlled by a gamepad.

More information about the inputs library can be found here: https://inputs.readthedocs.io/en/latest/

- Final_Yolo_Video: This code is the same as the Video_And_Motion code, but it doesn’t have the ability to control the motion of the drone. This code was used to test the compatibility of the drones video feature. 

- Possible Source of information: https://github.com/bkbilly/parrot-gamepad

import cv2
import tensorflow as tf



CATEGORIES = ["Dog", "Cat"]

def prepare(filepath):
    IMG_SIZE = 100  # 50 in txt-based
    image = cv2.imread(filepath) 
    # Checking if the image is empty or not 
    if image is None: 
        print("Image is empty!!") 
    else: 
        print("Image is not empty!!")
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) # return the image with shaping that TF wants.

# Loads the model. 
model = tf.keras.models.load_model('my_model.keras')
prediction = model.predict([prepare(r"C:\Users\garne\OneDrive\Documents\Dog image\Test.jpg")])
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])
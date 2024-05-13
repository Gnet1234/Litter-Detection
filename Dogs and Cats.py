import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle
import numpy as np

DATADIR = r"C:\Users\garne\OneDrive\Documents\Dogs and Cats\PetImages"
# Used to identiy the directory for the dataset.
CATEGORIES = ["Dog", "Cat"]
# Creates the different categores that the code will use. 

for category in CATEGORIES:
    path = os.path.join(DATADIR, category) # Iterates through the dataset and sorts them based on the catergory.
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) # convert to array
        # Uses the path to iterate through the dataset and store them into an array. The images would be turned black and white with the grayscale command.
        plt.imshow(img_array, cmap = "gray")
        #plt.show()
        # Shows the image in gray.
        break
    break

IMG_SIZE = 100

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # resize to normalize data size
plt.imshow(new_array, cmap='gray')
#plt.show()
# Resizes the image to 100 x 100.

training_data = []

def create_training_data():
    for category in CATEGORIES:  # do dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()

print(len(training_data))

random.shuffle(training_data)

# The purpose of this is to randomize the order of images so the model doesn't develop a bias or pattern that will affect it in the future. 
for sample in training_data[:10]:
    print(sample[1])
    
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
    
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)


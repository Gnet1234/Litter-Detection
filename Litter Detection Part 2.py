import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow import keras
import pickle

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0
# Normilization process, which is used to scale down the image and reduce the spread of pixels. 

NAME = "Litter"
# This is for tensoboard so it can show our data. 

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
early_stopper = EarlyStopping(monitor = 'accuracy', min_delta = 0.01, patience = 3, verbose = 1, mode = 'max')

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# callback is used in conjunction with training using model.fit() to save a model or weights (in a checkpoint file) at some interval, so the model or weights can be loaded later to continue the training from the state saved.
checkpoint_filepath = r"C:\Users\garne\OneDrive\Documents\Dogs and Cats\Checkpoint.keras"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='accuracy',
    mode='max',
    save_freq='epoch',
    save_best_only=True)
# save_best_only = True is used to control what model is saved based on the monitored value.
# Max contorls if the callback function is looking for the maximum or minimum monitored value.  

model.fit(X, y, batch_size=32, epochs=5, validation_split=0.3, callbacks=[model_checkpoint_callback, tensorboard, early_stopper])
# The validation split function, takes a portion of the data and dosen't use it to train. The loss of the model will be based on the data split from the validation split. 
#callbacks = [tensorboard], Use tensorboard --logdir=logs/ in the terminal after creating the model to get the tensorboard. It will create a link you can open on a browser. 

# Save the model
model.save('my_model.keras')
model.evaluate(X,y)
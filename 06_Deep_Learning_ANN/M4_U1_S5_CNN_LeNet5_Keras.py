# LeNet-5 in Keras
# Here's how we can implement this architecture using Keras API.

# The first step is to import the necessary libraries from TensorFlow/Keras.
# Importing the Keras libraries and packages
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import Sequential # For initializing NN
from tensorflow.keras.layers import Convolution2D # For convolution, 1er step
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
# For pooling, 2nd step
from tensorflow.keras.layers import Flatten # For flattening, 3rd step
from tensorflow.keras.layers import Dense # For adding fully-connected layers towards the output layer
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support

# The first part of CNN is then implemented, which is to define the necessary convolution, average
# pooling, and flattening steps, specifying the appropriate dimensions for filters and pooling, as well
# as defining the stride used and number of filters. We also specify the activation functions used
# after convolutions.
#============================================================================
# CNN setup
#============================================================================
# CNN initialization
model = Sequential()
# Step 1 - 1st Convolution
# At convolution: no filters, rows, columns.
model.add(Convolution2D(filters=6,kernel_size=(3, 3),activation='relu',input_shape=input_shape))
# Step 2 - 1st Avg. Pooling
model.add(AveragePooling2D(pool_size=(2, 2),strides=2))
# Step 3 - 2nd Convolution
model.add(Convolution2D(filters=16,kernel_size=(3, 3),activation='relu'))
# Step 4 - 2nd Avg. Pooling
model.add(AveragePooling2D(pool_size=(2, 2),strides=2))
# Step 5 - Flattening
model.add(Flatten())
# Finally, the fully-connected layer is defined as an ANN. We specify the two hidden layers with
# their dimensions, as well as the activation functions used. The penultimate step defines the output
# (number of classes) along with the softmax activation function. Finally, the model is compiled,
# where the cost function used is specified, as well as the optimization algorithm used and the
# metrics considered.
#============================================================================
# NN for classification (Fully Connected)
#============================================================================
# Input: n_batch x 120
# HL: 120 x 120
# Output: n_batch x 120
model.add(Dense(units=120, activation='relu'))
# Input: n_batch x 120
# HL: 120 x 84
# Output: n_batch x 84
model.add(Dense(units=84, activation='relu'))
# Input: n_batch x 84
# Output: Multiclass classification dor 10 categories
model.add(Dense(units=10, activation = 'softmax'))
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

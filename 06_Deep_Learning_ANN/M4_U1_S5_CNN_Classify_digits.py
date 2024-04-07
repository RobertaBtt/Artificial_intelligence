# In this way, we can apply this architecture to classify the digits of the MNIST dataset as follows.
# Importing the Keras libraries and packages
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import Sequential # For initializing NN
from tensorflow.keras.layers import Convolution2D # For convolution, 1er step
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
#
# For pooling, 2nd step
from tensorflow.keras.layers import Flatten # For flattening, 3rd step
from tensorflow.keras.layers import Dense # For adding fully-connected layers towards the output layer
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
height, width = (28, 28) # 28x28is the dimensionality of the data
batch_size = 128
num_classes = 10
epochs = 12
# Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], height, width, 1)
x_test = x_test.reshape(x_test.shape[0], height, width, 1)
input_shape = (height, width, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# turn class vectors into binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#============================================================================
# CNN setup
#============================================================================
# CNN initialization
model = Sequential()
# Step 1 - 1st Convolution
# At convolution: no filters, rows, columns.
model.add(Convolution2D(filters=6,
kernel_size=(3, 3),
activation='relu',
input_shape=input_shape))
# Step 2 - 1st Avg. Pooling
model.add(AveragePooling2D(pool_size=(2, 2),
strides=2))
# Step 3 - 2nd Convolution
model.add(Convolution2D(filters=16,
kernel_size=(3, 3),
activation='relu'))
# Step 4 - 2nd Avg. Pooling
model.add(AveragePooling2D(pool_size=(2, 2), strides=2))
# Step 5 - Flattening
model.add(Flatten())
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
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
# In the code snippet above, we loaded the MNIST data from Keras and trained LeNet-5 to predict
# the category to which the digits belong. Data input are images already expressed as a numerical
# matrix, except that they are at 28x28 pixels, rather than 32x32, so CNN input has been
# generalized to work with the input size that is defined.
# Next, we show from Keras the dimensions of the network layers. As we said before, in this case,
# there are no trainable parameters within the pooling or flattening steps. The number of parameters
# differ a little bit from the original archivtecture, since we are using a smaller kernel (3x3) instead
# of the original one (5x5). In any case, 3x3x1x6+6 equals the value provided by Keras (60).
model.summary()
# _________________________________________________________________
# Below, we show the system evaluation by using accuracy, precision, recall, and confusion matrix
# metrics. As we can see, the accuracy metric is extremely high, although more complex structures
# achieve even better results over MNIST.37
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Obtain predictions for all test set
y_pred = model.predict(x_test).round()
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit([1,2,3,4,5,6,7,8,9,10])
y_pred = lb.inverse_transform(y_pred)
y_test = lb.inverse_transform(y_test)
# Evaluate results
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ", cm)
precision, recall, fbeta, support = precision_recall_fscore_support(y_test, y_pred)
print("Precision: ")
print(precision)
print("Recall: ")
print(recall)
# Loss test: 0.0373
# Accuracy Test: 0.9897

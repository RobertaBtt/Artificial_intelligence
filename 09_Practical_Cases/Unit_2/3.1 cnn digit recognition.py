import keras
from keras.datasets import mnist
from keras import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation
from keras.activations import softmax
from keras.optimizers import Adadelta
from keras import backend as K
import tensorflow as tf

import matplotlib.pyplot as plt

# Load data and split to train and test
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#plot 2 images as grayscale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))


# set adjustment parameters of the model

batch_size = 128
num_classes = 10
epochs = 12


# flatten datset of images (28x28) to a vector with input values of 784 pixels
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# normalize grayscale pixel values, between 0 and 255 to a range between 0 and 1

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# Apply "active coding"
# convert the class vectors to binary class matrices
# digit "2" will convert the vector from [0 1 2 3 4 5 6 7 8 9] to [0 0 1 0 0 0 0 0 0 0]
# use built-in utils.to_categorical to do this

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# DEFINE CNN

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation = Activation(tf.nn.softmax)))

# Compile NN with ADADELTA as optimization algorithm

model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.Adadelta(),
              metrics=['accuracy'])



# train nn with subset of data of 128 and 12 iterations


model.fit(X_train, y_train, batch_size=batch_size, epochs = epochs, verbose=1, validation_data=(X_test, y_test))


#Evaluate model

# Evaluate the model on the test set
score = model.evaluate(X_test, y_test, verbose=0)

# Print the Test Loss and Test Accuracy
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])


import numpy as np

# Choose a random index from the test set
random_index = np.random.randint(0, len(X_test))

# Get the random image and its true class
random_image = X_test[random_index]
true_class = np.argmax(y_test[random_index])

# Reshape the image to match the input shape expected by the model
random_image = np.reshape(random_image, (1, img_rows, img_cols, 1))

# Make predictions using the trained model
predictions = model.predict(random_image)

# Get the predicted class
predicted_class = np.argmax(predictions)

# Plot the actual and predicted class
plt.imshow(random_image.squeeze(), cmap=plt.get_cmap('gray'))
plt.title(f"True Class: {true_class}, Predicted Class: {predicted_class}")
plt.show()






































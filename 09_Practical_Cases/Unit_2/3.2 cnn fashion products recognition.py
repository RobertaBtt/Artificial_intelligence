
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.utils import to_categorical
from keras.datasets import fashion_mnist

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, InputLayer, LeakyReLU

from sklearn.model_selection import train_test_split

# Load fashion dataset and train test sets
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

#find unique numbers from train labels

classes = np.unique(y_train)
num_classes = len(classes)

plt.figure(figsize=[5,5])

#display first image in training data
plt.subplot(121)
plt.imshow(X_train[0,:,:], cmap='gray')
plt.title('Ground Truth:{}'.format(y_train[0]))

#display first image in test data
plt.subplot(122)
plt.imshow(X_test[0,:,:], cmap='gray')
plt.title('Ground Truth: {}'.format(y_test[0]))
plt.show()

# Preprocess data
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255
X_test = X_test / 255


# Change labels from categorical to one hot encoding
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)


#train test split
X_train, X_valid, train_label, valid_label = train_test_split(X_train, y_train_hot, test_size=0.2, random_state=13)

# DEFINE CNN
batch_size = 64
epochs = 20
num_classes = 10
'''
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='linear', input_shape=(28,28,1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Conv2D(64, kernel_size=(3,3), activation='linear', input_shape=(28,28,1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Conv2D(128, kernel_size=(3,3), activation='linear', input_shape=(28,28,1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(num_classes, activation='softmax'))


#compile using ADAM

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# Train cnn
train = model.fit(X_train, train_label, batch_size=batch_size, epochs = epochs, verbose=1, validation_data=(X_valid, valid_label))

# Evaluate
test_eval = model.evaluate(X_test, y_test_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


# plot accuracy and loss between training and validation data

accuracy = train.history['accuracy']
val_accuracy = train.history['val_accuracy']
loss = train.history['loss']
val_loss = train.history['val_loss']

epochs = range(len(accuracy))


plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

'''
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='linear', input_shape=(28,28,1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3,3), activation='linear', input_shape=(28,28,1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3,3), activation='linear', input_shape=(28,28,1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))


#compile using ADAM

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# Train cnn
train = model.fit(X_train, train_label, batch_size=batch_size, epochs = epochs, verbose=1, validation_data=(X_valid, valid_label))

# Evaluate
test_eval = model.evaluate(X_test, y_test_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


# plot accuracy and loss between training and validation data

accuracy = train.history['accuracy']
val_accuracy = train.history['val_accuracy']
loss = train.history['loss']
val_loss = train.history['val_loss']

epochs = range(len(accuracy))


plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()




# make predictions and show result

predicted_classes = model.predict(X_test)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)

correct = np.where(predicted_classes==y_test)[0]
print('Found %d correct labels' % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title('Predicted {}, Class{}'.format(predicted_classes[correct], y_test[correct]))
    plt.tight_layout()
plt.show()


incorrect = np.where(predicted_classes!=y_test)[0]
print('Found %d incorrect labels' % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title('Predicted {}, Class{}'.format(predicted_classes[incorrect], y_test[incorrect]))
    plt.tight_layout()





































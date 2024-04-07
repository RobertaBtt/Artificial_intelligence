# Example exercise: Image classification with LeNet-5
# In addition to working with images already transformed to their numerical values, with Keras we
# can work directly with images in their original format (jpg,png...).
# As an example for this we will build a classification system for the data set of:
# https://www.kaggle.com/alxmamaev/flowers-recognition
# In that dataset we have the following:
# ▪ 4242 images of flowers.
# ▪ The images are separated into 5 categories corresponding to different types of flowers:
# Daisy, Dandelion, Rose, Sunflower and Tulip. Each of these categories has 800 images
# of the corresponding flower.
# ▪ The format of the images is 320x240 pixels.
# The first step is, again, we define the libraries to be used.
# Importing the Keras libraries and packages
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from shutil import copyfile, rmtree
from tensorflow.keras.models import Sequential # To initialize the NN
from tensorflow.keras.layers import Convolution2D # To do the convolution, 1st
step
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D # For
pooling, step 2
from tensorflow.keras.layers import Flatten # For flattening, step 3
from tensorflow.keras.layers import Dense # For adding the fully-connected
layers
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img,
img_to_array
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.metrics import f1_score, confusion_matrix, precision_score,
recall_score, auc, log_loss
Subsequently, since the source data is not separated into train/test, we created a function to read
the files and separate them into two different folders, one to train the model and one to do the
subsequent test.
def prepare_train_test_set(path_ref="",train_per=0.8):
 """
 Function that loads the original images and splits them into a train and
test
 folder according to the split percentage defined in the arguments.
 Parameters
 ----------
 path_ref : String, optional
 Relative path for the folder where the images are located and
 where the new fodlers are gonnna be created. The default is "".
 train_per : Float, optional
 Train set percentage. The default is 0.8.
 Returns
 -------
 None.
 """
 path_full = path_ref + '/flowers'
 entries = os.listdir(path_full + '/')
 os.makedirs(path_ref + '/train') # Crear carpeta de train
 os.makedirs(path_ref + '/test') # Crear carpeta de test
 rmtree(path_full + '/flowers') # Eliminar esta carpeta innecesaria

 for entry in entries:
 print("Preparing datasets...")
 print("Entry: ", entry)
 files_inside = os.listdir(path_full + '/' + entry + '/')
 np.random.shuffle(files_inside)
 len_train = np.int(np.round(train_per*len(files_inside)))
 len_test = len(files_inside) - len_train

 os.makedirs(path_ref + '/train' + '/' + entry)
 os.makedirs(path_ref + '/test' + '/' + entry)

 [copyfile(path_full + '/' + entry + '/' + file, path_ref + '/train/' +
entry + '/' + file) for file in files_inside[:len_train]]
 [copyfile(path_full + '/' + entry + '/' + file, path_ref + '/test/' +
entry + '/' + file) for file in files_inside[len_train:]]
The next step is to specify how to load the images into the script to then train the model. This
loading is done from the Keras ImageDataGenerator function, which allows us to build a generator
with all those images and then train the neural network.
On this function, we build a wrapper to adjust the parameters of it in the image upload according
to whether this data is the training or test data. For training we specify the percentage of them
that will be used in validation. Thanks to this general approach, with this function we define later
if we want the images to be loaded in grayscale or color, the batch size used...
def image_data_generator(data_dir="",
 train_data=False,
 batch_size=10,
 target_size=(100, 100),
 color_mode='rgb',
 class_mode='binary',
 shuffle=True):
 """
 Function to load the images and use them in the NN.

 Parameters
 ----------
 data_dir : String, optional
 Path where the images are located. The default is "".
 train_data : TYPE, optional
 Whether to load datata with a train preprocessing or load it raw.
 The default is False.
 batch_size : Integer, optional
 Batch size. The default is 10.
 target_size : Tuple, optional
 Dimensionality of the images (height, width). The default is (100, 100).
 color_mode : String, optional
 Color model used. The default is 'rgb'.
 class_mode : String, optional
 Class mode. 'categorical' for multiclass and 'binary' for binary.
 The default is 'binary'.
 shuffle : Boolean, optional
 Specifies whether to shuffle or not that input data. The default is
True.

 Returns
 -------
 generator : generator
 Generator with the images to use later on.
 """

 if train_data:
 datagen = ImageDataGenerator(rescale=1./255,
 # rotation_range=20,
# width_shift_range=0.2,
# height_shift_range=0.2,
 # shear_range=0.2,
 # zoom_range=0.2,
# horizontal_flip=True,
validation_split=0.2
)
 else:
 datagen = ImageDataGenerator(rescale=1./255)

 generator = datagen.flow_from_directory(data_dir,
 target_size=target_size,
 color_mode=color_mode,
 batch_size=batch_size,
 shuffle=shuffle,
 class_mode=class_mode)
 return generator
Although LeNet-5 is originally defined for specific parameter values (image size, channels...), we
can generalize the architecture for any number of channels or image size as follows.
def build_model(optimizer="adam",
 loss='binary_crossentropy',
 height=32,
 width=32,
 channels=1,
 output_classes=1,
 final_activation="sigmoid"):
 """
 Architecture for the CNN
 Convolutional #1
 Activation any activation function, we will relu
 Average Pooling #1

 Convolutional #2

 Activation any activation function, we will relu
 Average Pooling #2

 Flatten Flatten the output shape of the final pooling layer
 Fully Connected #1 outputs 120

 Activation any activation function, we will relu
 Fully Connected #2 outputs 84

 Activation any activation function, we will relu
 Fully Connected (Logits) #3 output output_classes
 Parameters
 ----------
 optimizer : String, optional
 Algorithm for optimization. The default is "adam".
 loss : String, optional
 Loss function used. The default is 'binary_crossentropy'.
 height : String, optional
 Image height. The default is 32.
 width : String, optional
 Image width. The default is 32.
 channels : Integer, optional
 Number of channels used. The default is 1.
 output_classes : Integer, optional
 Number of output classes. The default is 1.
 final_activation : String, optional
 Final activation function. 'sigmoid' for binary, 'softmax' for
multiclass.
 The default is "sigmoid".
 Returns
 -------
 model : Object
 Trained model.
 """
 # CNN inittialization
 model = Sequential()

 # Step 1 - Convolution
 # For convolution: nº filters, rows, columns.
 # We also define the kernel dimension (same for all channels)
 model.add(Convolution2D(filters=6,
 kernel_size=(3, 3),
 padding='same',
 activation='relu',
 input_shape=(height,width,channels)))


 # Step 2 - 1er Avg. Pooling
 # The size of the kernel for avg. pooling is 2x2
 model.add(AveragePooling2D(pool_size=(2, 2),
 strides=2))

 # Step 3 - 2n Convolution
 model.add(Convolution2D(filters=16,
 kernel_size=(3, 3),
 padding='valid',
 activation='relu'))

 # Step 4 - 2nd Avg. Pooling
 model.add(AveragePooling2D(pool_size=(2, 2),
 strides=2))

 # Step 5 - Flattening
 model.add(Flatten())

 # Step 6 - Fully connected layers
 model.add(Dense(units=120, activation='relu'))
 model.add(Dense(units=84, activation='relu'))

 # Output
 model.add(Dense(units=output_classes,
 activation = final_activation))

 # Model compile
 model.compile(loss=loss,
 optimizer=optimizer,
 metrics=['accuracy'])

 return model
This will allow us to use LeNet-5 on a larger set of color images than the original study, as well
as to be able to define a number of different output classes.
Finally, we develop the entire execution pipeline. First, loading the data and running the train/test
data separation function.
# ========================================================================
# 1. Setup
# ========================================================================
# Parameters (I)
tf.random.set_seed(42)
path_ref = 'datasets/flowers-recognition'
prepare_train_test_set(path_ref=path_ref ,train_per=0.8)
# Parameters (II)
batch_size = 400
height, width = (240, 320)
epochs = 10
color_mode = "rgb" # 'rgb' or 'grayscale'
optimizer = "adam"
loss = "categorical_crossentropy"
class_mode="categorical"
path_train = path_ref + '/train'
path_test = path_ref + '/test'
output_classes=5 # Number of output classes
final_activation="softmax"
# Channels according to color mode
if color_mode == "grayscale":
 channels = 1
 grayscale = True
else:
 channels = 3
 grayscale = False
# Visualize sample image
img = load_img(path_train + '/daisy/5673551_01d1ea993e_n.jpg',
 target_size = (height, width),
 grayscale=grayscale)
imgplot = plt.imshow(img)
# Load images from train/test train/test
train_generator = image_data_generator(path_train,
 train_data=True,
 batch_size=batch_size,
 target_size=(height, width),
 color_mode=color_mode,
 class_mode=class_mode,
 shuffle=True)
test_generator = image_data_generator(path_test,
 train_data=False,
 batch_size=batch_size,
 target_size=(height, width),
 color_mode=color_mode,
 class_mode=class_mode,
 shuffle=True)
After that, we train CNN. We have specified a batch size and a relatively high number of epochs
for good results. This, along with the size of the images will cause the model to take a long time
to train (it could reach several hours depending on the input parameters used). For this reason,
it is essential to save the model locally once trained so that we do not have to repeat the training
unless necessary.
# ========================================================================
# 2. Training
# ========================================================================
# Model definition & visualization of the specified architecture
model = build_model(optimizer=optimizer,
 loss=loss,
 height=height,
 width=width,
 channels=channels,
 output_classes=output_classes,
 final_activation=final_activation)
print(model.summary())
# Model fitting & training
model.fit_generator(train_generator,
 steps_per_epoch=3458//batch_size,
 epochs=epochs)
# GSave model as binary file
model.save('model_flowers.h5')
Finally, we evaluate the results on the test images. To do this, as we see in the code, we can
evaluate individual images or see the metrics used during training but evaluated on the test data.
Similarly, to see the other metrics, what we do is load the images, express them in their numeric
format, and get the predictions of the model based on those input numeric arrays. To do this we
read the test folders and create the y_test vector, since we know the category to which the images
belong. Then, we pass them to a numeric matrix, and evaluate the model on them. Finally, we
compare the final two outputs of the test vectors and the predicted one.
# ========================================================================
# 3. Evaluation
# ========================================================================
# Analize predictions for test images
test_image = load_img(path_test + '/rose/110472418_87b6a3aa98_m.jpg',
 target_size = (height, width),
 grayscale=grayscale)
imgplot = plt.imshow(test_image)
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print("Prediction: ", result)
test_image = load_img(path_test + '/sunflower/35477171_13cb52115c_n.jpg',
 target_size = (height, width),
 grayscale=grayscale)
imgplot = plt.imshow(test_image)
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print("Prediction: ", result)
# Loss/Accuracy
score = model.evaluate(test_generator, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Transform test matrices into numeric arrays
entries = os.listdir(path_test)
X_test = []
y_test = []
for entry in entries:
 subpath = path_test + '/' + entry
 files = []
 for _, _, f in os.walk(subpath):
 files += f

 files = [x for x in files if 'jpg' in x]
 X_test += [np.expand_dims(img_to_array(load_img(subpath + '/' + f,
 target_size = (height,
width),
 grayscale=grayscale)),
axis = 0) for f in files]
 if entry == "daisy":
 y_test += [0]*len(files)
 elif entry == "dandelion":
 y_test += [1]*len(files)
 elif entry == "rose":
 y_test += [2]*len(files)
 elif entry == "sunflower":
 y_test += [3]*len(files)
 elif entry == "tulip":
 y_test += [4]*len(files)
# Obtaining predictions for all test data
y_pred = [model.predict_classes(x)[0] for x in X_test]
y_pred = []
for x in X_test:
 y_pred.append(model.predict_classes(x)[0])
# Evaluate results
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ", cm)
print("Precision: ", np.round(precision_score(y_test, y_pred,
average='macro'),2))
print("Recall: ", np.round(recall_score(y_test, y_pred, average='macro'),2))
print("f1_score: ", np.round(f1_score(y_test, y_pred, average='macro'),2))
With all this, we get the following results. As we can see, the model fails to reach very high metrics,
and although it seems that certain classes are well distinguished, in some others the rate of
correct predictions compared to their false negatives is low.
Illustration 64 Metrics
Illustration 65 Confusion Matri
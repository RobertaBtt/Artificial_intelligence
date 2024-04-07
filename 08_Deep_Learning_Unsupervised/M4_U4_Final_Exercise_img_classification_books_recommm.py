# 1. FINAL EXERCISE
# This final exercise is divided into two independent sections, one related to the supervised model’s
# part, and the other related to the unsupervised model’s part.
# The first part aims to apply the CNN algorithm to image classification, while the second part aims
# to apply the RBM algorithm to recommender systems.
#
# --------------
# 1.1 Part 1: Image Classification
# For this section we will use the Kaggle dataset from https://www.kaggle.com/sanikamal/horses-
# or-humans-dataset, which contains images of horses and human beings. With this, similar to the
# example exercise within the course, it is requested to build a CNN for classification that uses this
# data using LeNet-5 as reference architecture.
# This dataset does not need to be separated in train / test since it is already separated from the
# source. The ‘train’ folder should be used to train and validate the model, and the ‘validate’ folder
# to do the final test.
# The model is a binary classification model, so this must be taken into account when specifying
# certain parameters (output function, output dimension ...). To do this, you can use the code in the
# notes as a reference if you want and modify the relevant parts for this problem.
# As a totally optional part, if you want, you can try with other of the architectures mentioned in the
# course to compare the results.
#
# ----------
# 1.2 Part 2: Recommender Systems
# The task to be carried out is to build a RecSys analogous to the one in the course, but this time
# to recommend books.
# For this, we are going to use the Kaggle dataset
# https://www.kaggle.com/zygmunt/goodbooks-10k, which is very similar in all respects to what we
# have seen from MovieLens, except that the ratings are about books .
# Since the set of books and users is very large (ratings.csv has almost 1M records), we are going
# to build a simple version of RecSys that only considers the first 1000 books that appear in
# books.csv. In this way we avoid computing problems that we would have if we used all the data.
#
#
# For this example, we ask you to do the implementation with PyTorch. We also ask you to carry
# out a small EDA of the input data. Finally, there are some questions that we want you to answer
# based on the data. Do the predictions for different users change a lot? Briefly comment on the
# results obtained. What books are recommended to user 1 that are not recommended to user 50?


2.1 Part 1: Image Classification
For this first part of the exercise, in the first place, we define the functions and libraries that we
will use later. We thus define the general function to instantiate a network with the LeNet-5
architecture, together with a function to load the images.
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 22:24:11 2020
@author: alber
"""
# Importing the Keras libraries and packages
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import Sequential # To initialize the NN (as it is
a Sequence of layers, We do it the same as with ANN; we do not use the initial
of Graph)
from tensorflow.keras.layers import Convolution2D # To do the convolution step,
1st step
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D # For the
Pooling step, step 2
from tensorflow.keras.layers import Flatten # For flattening, step 3
from tensorflow.keras.layers import Dense # To add the fully-connected layers
to the output layer


from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img,
img_to_array
from tensorflow.keras.layers import Dense, Dropout, Flatten
from
sklearn.metrics
import
f1_score,
confusion_matrix,
precision_score,
recall_score, auc, log_loss
# =========================================================================
# Part 1 - Building the CNN - (no need to do preprocessing in this case)
# =========================================================================
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
Final
activation
function.
'sigmoid'
for
binary,
'softmax'
for
multiclass.


The default is "sigmoid".
Returns
-------
model : Object
Trained model.
"""
# CNN initialization
model = Sequential()
# Step 1 - 1st Convolution
# In Convolution: no. Filters, rows, columns.
# The kernel dimension is also defined (same for all channels)
model.add(Convolution2D(filters=6,
kernel_size=(3, 3),
padding='same',
activation='relu',
input_shape=(height,width,channels)))
# Step 2 - 1st Avg. Pooling
# The kernel size of the avg. pooling is 2x2
model.add(AveragePooling2D(pool_size=(2, 2),
strides=2))
# Step 3 - 2nd Convolution
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
model.add(Dense(units = output_classes,
activation = final_activation))
# Model compile
model.compile(loss=loss,
optimizer=optimizer,
metrics=['accuracy'])
return model


Because it is a binary classification problem, when specifying the hyperparameters, we indicate
that we are going to use the sigmoid activation function for the output, that the number of classes
is 1 (a single binary variable), and that the cost function will be the cross entropy for the binary
case. Along with this, we define the rest of the hyperparameters for the exercise.
batch_size = 20
height, width = (32, 32)
epochs = 80
color_mode = "rgb"
optimizer = "adam"
loss = "binary_crossentropy"
class_mode='binary'
output_classes=1 # Number of output classes
final_activation="sigmoid"
After that, we proceed to load the data from the data set mentioned previously. This dataset is
already separated in train / validation / test. The training set consists of 1027 artificial images of
horses or people, and the objective of the model will be to distinguish between these images and
classify them into those two categories. Within that training set there are 527 images of people
and 500 of horses, so the dataset is quite balanced, and therefore it is not expected that there will
be imbalance problems.
As an illustration, we can load and view one of the images from Python as follows:
# View one image
# Numeric array (8 bits for each of R, G, B)
relative_path = 'datasets/horses-or-humans-dataset/horse-or-human'
img = mpimg.imread('{0}/train/horses/horse01-0.png'.format(relative_path))
print(img)
imgplot = plt.imshow(img)


We continue to load the 3 data sets, specifying the number of pixels that each of the images will
have:
# Load data from train/test
training_set = image_data_generator('{0}/train'.format(relative_path),
train_data=True,
batch_size=batch_size,
target_size=(height, width),
color_mode=color_mode,
class_mode=class_mode,
shuffle=True)
val_set = image_data_generator('{0}/validation'.format(relative_path),
train_data=False,
batch_size=batch_size,
target_size=(height, width),
color_mode=color_mode,
class_mode=class_mode,


shuffle=True)
test_set = image_data_generator('{0}/test'.format(relative_path),
train_data=False,
batch_size=batch_size,
target_size=(height, width),
color_mode=color_mode,
class_mode=class_mode,
shuffle=True)
Next, we instantiate the NN object, and train it using the training data set, and obtaining the error
metrics in the different epochs against the validation data set.
# Definición del modelo y visualización de la arquitectura definida.
model = build_model(optimizer=optimizer,
loss=loss,
height=height,
width=width,
channels=channels,
output_classes=output_classes,
final_activation=final_activation)
print(model.summary())
# Definition of the model and visualization of the architecture.
model.fit_generator(training_set,
steps_per_epoch=batch_size,
epochs=epochs,


validation_data=val_set)
# We save the model in a binary file
model.save('model_horses_vs_humans.h5')
As a last step, we obtain the predictions on the test data set, and we calculate different metrics
on them:
# Check predictions for one image
test_image = load_img('{0}/test/horses/horse1-539.png'.format(relative_path),
target_size = (height, width))
test_image
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
test_image
=
load_img('{0}/test/humans/valhuman02-
12.png'.format(relative_path), target_size = (height, width))
test_image
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
if result[0][0] == 1:
prediction = 'human'
else:
prediction = 'horse'


# Loss/Accuracy
score = model.evaluate(test_set, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Load test set and turn it into matrices arrays
path = 'datasets/horses-or-humans-dataset/horse-or-human/test/'
entries = os.listdir(path)
X_test = []
y_test = []
for entry in entries:
subpath = path + entry
files = []
for _, _, f in os.walk(subpath):
files += f
X_test
+=
[np.expand_dims(img_to_array(load_img(subpath
+
'/'
+
f,
target_size = (height, width))), axis = 0) for f in files]
if entry == "horses":
y_test += [0]*len(f)
else:
y_test += [1]*len(f)
# Obtain predictions for all test set
y_pred = [model.predict_classes(x)[0][0] for x in X_test]


# Evaluate results
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(cm)
print("Precision: ", np.round(precision_score(y_test, y_pred),4))
print("Recall: ", np.round(recall_score(y_test, y_pred),4))
print("f1_score: ", np.round(f1_score(y_test, y_pred),4))
As stated in the exercice, we will try different configurations to see the variation in the results:
▪Batch_size=20, epochs=80, optimizar=adam, color_mode=rgb
▪Batch_size=10, epochs=10, optimizar=adam, color_mode=rgb
▪Batch_size=20, epochs=10, optimizar=adam, color_mode=rgb


▪Batch_size=20, epochs=10, optimizar=adam, color_mode=rgb, size=(16,16)
▪Batch_size=20, epochs=10, optimizar=adam, color_mode=rgb, size=(8,8)
▪Batch_size=20, epochs=10, optimizar=adam, color_mode=rgb, grayscale
▪Batch_size=20, epochs=10, optimizar=sgd, color_mode=rgb


▪
Batch_size=20, epochs=10, optimizar= rmsprop, color_mode=rgb
In these evaluations we have considered different optimization algorithms (sgd, rmsprop, adam),
along with various batch_size sizes, number of epochs, and also considering different image sizes
and whether or not to use grayscale or use the color image. The idea is to find the combination
that gives the best results while being as simple as possible, thus reducing the computational cost
as much as possible. Apparently using Adam we get consistently better results. And using a
batch_size of 20 and 10 epochs seems to be enough to reach the optimum. These results are
even better if we use grayscale instead of rgb, so this combination seems to be the best among
the ones analyzed.
2.2 Parte 2: Recommender Systems
For this section, it is requested to build a recommender system (RecSys) of books, using the
dataset mentioned in the statement.
This data set, in its original form, consists of ratings on a set of 10,000 books by 6,248 users, who
give ratings on a scale of 1 to 5, depending on whether they liked the book more or less.
Before proceeding with the exercise itself, we define the RBM class in a manner analogous to
what was done in the course. We also import the relevant libraries for this exercise.
# Boltzmann Machines
# Importing the libraries
import numpy as np
import pandas as pd


import torch
import torch.nn.parallel
import torch.utils.data
# =========================================================================
# 0. Define RBM
# ========================================================================
# Creating the architecture of the Neural Network
class RBM():
def __init__(self, nv, nh, batch_size, nb_epoch, k_steps, learning_rate,
verbose):
self.W = torch.randn(nh, nv)
self.a = torch.randn(1, nh)
self.b = torch.randn(1, nv)
self.nh = nh
self.nv = nv
self.verbose = verbose
self.batch_size = batch_size
self.nb_epoch = nb_epoch
self.k_steps = k_steps
self.learning_rate = learning_rate
def sample_h(self, x):
wx = torch.mm(x, self.W.t())
activation = wx + self.a.expand_as(wx)
p_h_given_v = torch.sigmoid(activation)
return p_h_given_v, torch.bernoulli(p_h_given_v)


def sample_v(self, y):
wy = torch.mm(y, self.W)
activation = wy + self.b.expand_as(wy)
p_v_given_h = torch.sigmoid(activation)
return p_v_given_h, torch.bernoulli(p_v_given_h)
def update_weights(self, v0, vk, ph0, phk):
learning_rate = self.learning_rate
self.W
+=
learning_rate*(torch.t(torch.mm(v0.t(),
ph0)
-
torch.mm(vk.t(), phk)))
self.b += learning_rate*(torch.sum((v0 - vk), 0))
self.a += learning_rate*(torch.sum((ph0 - phk), 0))
def train(self, training_set):
batch_size = self.batch_size
nb_epoch = self.nb_epoch
k_steps = self.k_steps
verbose = self.verbose
# Training the RBM
for epoch in range(1, nb_epoch + 1):
train_loss = 0
s = 0.
nb_users = len(training_set)
for id_user in range(0, nb_users - batch_size, batch_size):
vk = training_set[id_user:id_user+batch_size]
v0 = training_set[id_user:id_user+batch_size]
ph0,_ = self.sample_h(v0)


for k in range(k_steps):
_,hk = self.sample_h(vk)
_,vk = self.sample_v(hk)
vk[v0<0] = v0[v0<0]
phk,_ = self.sample_h(vk)
self.update_weights(v0, vk, ph0, phk)
train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
s += 1.
if verbose:
print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
def evaluate(self, test_set):
verbose = self.verbose
nb_users = len(test_set)
# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
v = training_set[id_user:id_user+1]
vt = test_set[id_user:id_user+1]
if len(vt[vt>=0]) > 0:
_,h = self.sample_h(v)
_,v = self.sample_v(h)
test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
s += 1.
if verbose:
print('test loss: '+str(test_loss/s))


return test_loss/s
def predict(self, v_user):
_,h = self.sample_h(v_user)
_,v = self.sample_v(h)
return v
After that, we load the dataset. As we can see, an important point is to filter the book catalog to
use only the first 1000, as mentioned in the beginning.
# =========================================================================
# 1. Load & prepare data
# =========================================================================
# Load data
path_relative = "datasets"
df_books = pd.read_csv(path_relative + '/goodbooks-10k/books.csv')
df_books = df_books[df_books['id']<=1000]
df_ratings = pd.read_csv(path_relative + '/goodbooks-10k/ratings.csv')
df_ratings
df_ratings[df_ratings['book_id'].isin(df_books['id'])][['user_id',
=
'book_id',
'rating']]
print(df_ratings.head())
As we can see, indeed, the ratings of different users appear on the existing books in the catalog.


As the exercise mentions, we are going to carry out a small EDA of the data. Although different
aspects can be covered, an interesting one is to know the distribution of user ratings. We get the
following:
It can be seen how most of the user ratings are above 3. This can give us information on what
value to consider as the threshold for "I like" versus "I don't like". We will keep the case of
considering the difference in 3, as in the course.
With the previous dataframe, the matrix is pivoted in order to have a sparse matrix in which each
user represents a row and each item (book) a column, with the ratings as the value of the cells.
When an item has not been rated by a user, it will be represented as -1.
# Pivot table
df_input = pd.pivot_table(df_ratings, values='rating', index=['user_id'],
columns=['book_id'], aggfunc=np.sum)
# Deal with NaN
df_input = df_input.fillna(-1)
print(df_input.head())


We continue making a separation between the data that we will use to train the RBM and those
that we will use for the evaluation phase (which has never seen the model and on which we will
try to predict the ratings, to see if they match the real ones). These matrices are expressed as
PyTorch tensors, and then their original value is converted to a binary value that reflects whether
or not they liked the book.
# Train/Test split
X = df_input.sample(frac=1).round() # Round ratings
len_train = int(np.round(0.8*len(X)))
X_train = X[:len_train]
X_test = X[len_train:]
# Converting to PyTorch tensors
training_set = X_train.values
test_set = X_test.values
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)
# Converting the ratings into binary ratings 1 (Liked) or 0 (Did not liked)
training_set[training_set == 1] = 0


training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1
We define the hyperparameters that we are going to consider from the RBM, we instantiate the
class object and train the model.
# ========================================================================
# 2. Train RBM
# ========================================================================
# Hyperparameters
nv = len(training_set[0])
nh = 100
batch_size = 100
nb_epoch = 10
k_steps = 10
learning_rate = 1
verbose = True
# Train model
rbm = RBM(nv, nh, batch_size, nb_epoch, k_steps, learning_rate, verbose)
rbm.train(training_set)
Finally, we evaluate the model by comparing the reconstructed vectors for the test users against
their real ratings, in order to see if they match or not.


# ========================================================================
# 3. Evaluate and obtain predictions
# ========================================================================
# Evaluation
rbm.evaluate(test_set)
# Obtain an individual prediction
v_test = test_set[0:1]
v_pred = rbm.predict(v_test)
# Check book for that prediction
id_books = ([v_pred==1][0][0]) & ([v_test==-1][0][0])
id_books = id_books.tolist()
# Combine with book titles and see results for units with missing values
df_recom = pd.DataFrame({'id_check':id_books})
df_recom
=
df_recom.reset_index().rename(columns={'index':'book_id'}).merge(df_books)
print("Recommended books: ")
print(df_recom[df_recom['id_check']==True]['title'].head(10))
We obtain a value for the cost function of 0.1804.


On the other hand, as an example, we visualize the prediction for one of test users. To do this,
we obtain the identifiers of the books that are not rated (value -1 in the original vector) and that
have a value of 1 in the predicted one, resulting in the following:
Regarding the last question of the exercise, we are going to check if the predictions change a lot
between users.
list_recom_books = []
for i in range(len(test_set)):
v_pred = rbm.predict(test_set[i:i+1])
n_recom = sum(v_pred.tolist()[0])
list_recom_books.append(n_recom)
print("Median of recommended books: ", np.median(list_recom_books))
We observe that the median number of recommended books is 897.7. In this way, our system
tends to recommend a large part of the initial 1000 books to all users, leaving only a small part of
the set without recommending. This may be due to several reasons, one of them to the distribution
of the data, as we have seen at the beginning of the exercise: most users rate with values greater
than 3, so the system, as it is specified, would tend to recommend books in a general way.
It is not the object of the exercise, but it could be investigated by changing that cut-off value,
comparing the system metrics, and seeing if it is more interesting to consider only films with very


high scores (4.5 or only 5) as films”liked”. Finally, another factor that influences which books
appear as the most recommended is that our base of 1000 books is not very extensive. Using all
the books from the original set the RecSys could be enriched.
Analyzing the average score of the different books, we observe the following:
Indeed, some of the most popular books (highest average score) appear among the
recommended books for that example user that we have seen before. Each of the books has
been rated by 100 users. This number is much lower than the total number of existing users, but,
as we can see for book_id = 1 (“Harry Potter and the Half-Blood Prince”), most users would rate
it with a “like”. In this way it is a book that is usually going to be recommended.


Finally, we answer the question: "What books are recommended to user 1 that are not
recommended to user 50".
29
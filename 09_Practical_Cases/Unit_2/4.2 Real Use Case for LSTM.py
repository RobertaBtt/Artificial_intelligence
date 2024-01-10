import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error



# load data and visualize time series

df = pd.read_csv('00-Support Material/datasets/win/airline-passengers.csv', usecols=[1], engine='python')
dataset = df.values
dataset = dataset.astype('float32')


# Visualize the time series
plt.figure(figsize=(10, 6))
plt.plot(dataset, label='Airline Passengers')
plt.title('Airline Passengers Time Series')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.legend()
plt.show()

#normalize dataset
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

# train/test split
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# define function to take two arguments:  dataset (numpy array) and look_back (number of previous steps to use as input variables to predict next thing) default is 1
#the default will create a dataset where x is the number of passengers at a given time (t)
# and y is the numbe ro fpassengers at the next time (t + 1)

def create_dataset( dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


#use function to prep the train adn test data sets for modeling

look_back = 1
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

# the lstm network expects the input data (X) provided with a specific matrix structure in form of [samples, time steps, features].

# the data is in the form [samples, features]

#transform the prepared training and test inputs in the expteced structure using np.reshape()

X_train = np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0],1,X_test.shape[1]))


#now ready to design and adapt lstm network

batch_size = 1
epochs = 100

# Create LSTM
model = Sequential()
model.add(LSTM(4, input_shape = (1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, verbose=2)


# ake predictions
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

#invert rpedictions
predict_train = scaler.inverse_transform(predict_train)
y_train = scaler.inverse_transform([y_train])

predict_test = scaler.inverse_transform(predict_test)
y_test = scaler.inverse_transform([y_test])


# calculate rms error
score_train = math.sqrt(mean_squared_error(y_train[0], predict_train[:,0]))
print('Train Score: %.2f RMSE' % (score_train))

score_test = math.sqrt(mean_squared_error(y_test[0], predict_test[:,0]))
print('Test Score: %.2f RMSE' % (score_test))

# generate predictions using the model for both train and test datasets to get a visual indication of model's abilities
#because of how the dataset was prepped, we have to change the predictions so that they align on the x-axis with the original dataset


#shift train predictions for plotting
train_pred_plot = np.empty_like(dataset)
train_pred_plot[:,:] = np.nan
train_pred_plot[look_back:len(predict_train)+look_back,:] = predict_train

#shift test predictions for plotting
test_pred_plot = np.empty_like(dataset)
test_pred_plot[:,:] = np.nan
test_pred_plot[len(predict_train)+(look_back*2)+1:len(dataset)-1,:] = predict_test

#plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(train_pred_plot)
plt.plot(test_pred_plot)
plt.show()















































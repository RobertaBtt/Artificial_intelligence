import numpy as np
import scikeras

from sklearn.metrics import accuracy_score

from keras.datasets import reuters
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation, LSTM
from keras import optimizers
from scikeras.wrappers import KerasClassifier

#RNN params
epochs = 20
batch_size = 50


# Load data
num_words = 30000
maxlen = 50
test_split = 0.3

(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=num_words, maxlen=maxlen, test_split=test_split)

# Pretreat data obtained to use in RNN
# Pad sequence w zeros
# padding param set to 'post' => 0's are appended to end of sequences
X_train = pad_sequences(X_train, padding = 'post')
X_test = pad_sequences(X_test, padding = 'post')

X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1],1)) 
X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1],1)) 

y_data = np.concatenate((y_train, y_test))
y_data = to_categorical(y_data)

y_train = y_data[:1395]
y_test = y_data[1395:]

# Define model, compile it, train it, calculate prediction

def rnn():
    model = Sequential()
    model.add(SimpleRNN(50, input_shape=(49,1), return_sequences=False))
    model.add(Dense(46))
    model.add(Activation('softmax'))
    
    adam = optimizers.Adam(learning_rate = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer=adam, metrics = ['accuracy'])
    
    return model


model = KerasClassifier(model = rnn(), epochs = epochs, batch_size=batch_size, verbose=0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Show evaluation with obtained accuracy of []
y_test_ = np.argmax(y_test, axis=1)
y_pred_ = np.argmax(y_pred, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test_, y_pred_)

# Show the obtained accuracy
print(f"RNN Accuracy: {accuracy * 100:.2f}%")



# Stacked RNN
def stacked_rnn():
    model = Sequential()
    model.add(SimpleRNN(50, input_shape=(49,1), return_sequences=True))  #return sequence param is true so that it can be stacked)
    model.add(SimpleRNN(50, return_sequences=False))
    model.add(Dense(46))
    model.add(Activation('softmax'))
    
    adam = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    return model

model = KerasClassifier(model = stacked_rnn, epochs = epochs, batch_size=batch_size, verbose=0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Show evaluation with obtained accuracy of []
y_test_ = np.argmax(y_test, axis=1)
y_pred_ = np.argmax(y_pred, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test_, y_pred_)

# Show the obtained accuracy
print(f"Stacked Accuracy: {accuracy * 100:.2f}%")



# LSTM
def lstm():
    model = Sequential()
    model.add(LSTM(50, input_shape=(49,1), return_sequences=False))  
    model.add(Dense(46))
    model.add(Activation('softmax'))
    
    adam = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    return model

model = KerasClassifier(model = lstm, epochs = epochs, batch_size=batch_size, verbose=0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Show evaluation with obtained accuracy of []
y_test_ = np.argmax(y_test, axis=1)
y_pred_ = np.argmax(y_pred, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test_, y_pred_)

# Show the obtained accuracy
print(f"LSTM Accuracy: {accuracy * 100:.2f}%")





# STacked LSTM
def stacked_lstm():
    model = Sequential()
    model.add(LSTM(50, input_shape=(49,1), return_sequences=True))  
    model.add(LSTM(50, return_sequences=False))  
    model.add(Dense(46))
    model.add(Activation('softmax'))
    
    adam = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    return model

model = KerasClassifier(model = stacked_lstm, epochs = epochs, batch_size=batch_size, verbose=0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Show evaluation with obtained accuracy of []
y_test_ = np.argmax(y_test, axis=1)
y_pred_ = np.argmax(y_pred, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test_, y_pred_)

# Show the obtained accuracy
print(f"Stacked LSTM Accuracy: {accuracy * 100:.2f}%")












































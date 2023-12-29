# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:39:49 2020

@author: alber
"""
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import gensim.downloader as api

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, log_loss
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# 0. General functions
# =============================================================================
def word_vector(df_input, lemmatizer, word_vectors, vocabulary, col_sentences):
    """
    Function to preprocess the input words and get a list with
    The embeddings arrays of the words in each record.

    Parameters
    ----------
    df_input : dataframe
        input dataframe with all texts.
    lemmatizer : object
        NLTK stemming object.
    word_vectors : object
        Object with the word2vecs of the Gensim vocabulary.
    vocabulary : list
        list of existing words in Gensim's vocabulary.
    col_sentences : str
        column of the dataframe where the phrases are.

    Returns
    -------
    X : list
        List of lists in which each record has the list with the arrays
        of the embeddings of the words of that phrase. That is, X[0] has
        a list where each element corresponds to the embeddings of a word.
        Thus, for example, X[0][2] will be a vector of dimension 100 where it appears
        the vector of embeddings of the third word of the first sentence.
    """
    
    
    X = []
    
    for text in df_input[col_sentences]:
        
        # Tokenize every phrase
        words = re.findall(r'\w+', text.lower(),flags = re.UNICODE) # Paso a minusculas todo
        # Elimination of stop_words
        words = [word for word in words if word not in stopwords.words('english')]
        # Remove hyphens and other weird symbols
        words = [word for word in words if not word.isdigit()] # Elimino numeros    
        # Stemming 
        words = [lemmatizer.lemmatize(w) for w in words]
        # Delete words that are not in the vocabulary
        words = [word for word in words if word in vocabulary]
        # Word2Vec
        words_embeddings = [word_vectors[x] for x in words] 
            
        # Save the final sentence
        X.append(words_embeddings) # save as a numpy array
        
    return X


def create_RNN(x_train, K, n_lstm=8, loss='categorical_crossentropy', optimizer='adam'):
    """
    Function to create the RNN. As input parameter we only need the array
    of features to specify the input dimensionality of the NN.

    Parameters
    ----------
    x_input : array
       Matrix of input features.
    K: int
       Exit classes
    n_lstm : int, optional
        Number of lstm used. The default is 8.
    loss : string, optional
        loss metric. The default is 'categorical_crossentropy'.
    optimizer : string, optional
        optimizer. The default is 'adam'.

    Returns
    -------
    model : object
        Trained model.
    """
    
    # Begin sequence
    model = tf.keras.Sequential()
    
    # Add a LSTM layer with 8 internal units.
    model.add(LSTM(n_lstm, input_shape=x_train.shape[-2:]))
    
    # Add Dropout
    # model.add(Dropout(0.5))
    
    # # Another layer
    # model.add(Dense(100, activation='relu'))
    
    # # Output
    model.add(Dense(K, activation='sigmoid'))
    
    # Compile model
    model.compile(loss=loss, optimizer=optimizer)
    
    return model


# =============================================================================
# 1. Load Data
# =============================================================================
# Load files
tf.random.set_seed(42)
path_files = "predict-disaster"
df_raw = pd.read_csv(path_files+'/train.csv', encoding = "latin-1")
df_raw = df_raw[['text', 'target']]

# path_files = "datasets/sentiment-review"
# df_raw = pd.read_csv(path_files+'/train/train.tsv', sep='\t')

# Shuffle input
df_raw = df_raw.sample(frac=1)

# Load word2vec
word_vectors = api.load("glove-wiki-gigaword-100")
vocabulary = [x for x in word_vectors.vocab]

# Set lemmatizer
lemmatizer = WordNetLemmatizer() 

# X/y split
X = pd.DataFrame(df_raw['text'])
y = df_raw['target']

# =============================================================================
# 2. Preprocess
# =============================================================================
# Obtain X variable and prepare y.
X = word_vector(X, 
                lemmatizer,
                word_vectors, 
                vocabulary,
                col_sentences="text")

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Obtain tensor: [N_SENTENCES x SEQ_LENGTH x EMBEDDING_FEATURES]
SEQ_LENGTH = np.int(np.round(np.percentile([len(x) for x in X], 99, interpolation = 'midpoint')))
# SEQ_LENGTH = np.int(np.round(np.percentile([len(x) for x in X], 100, interpolation = 'midpoint')))
data_train = pad_sequences(X_train, 
                           maxlen=SEQ_LENGTH,
                           padding="post", 
                           truncating="post")

data_test = pad_sequences(X_test, 
                          maxlen=SEQ_LENGTH,
                          padding="post", 
                          truncating="post")

# =============================================================================
# 3. Train model
# =============================================================================
# Params
K = 1 # N classes
batch_size = 50
epochs = 5

# Create RNN
model = create_RNN(x_train = data_train,
                   K = K,
                   n_lstm = 10,
                   loss = 'binary_crossentropy',
                   optimizer = 'adam')
print(model.summary())

# Fit model
model.fit(data_train,
          y_train,
          epochs = epochs, 
          batch_size = batch_size)

# Save model
model.save('model_nlp_disaster.h5')

# =============================================================================
# 4. Evaluate
# =============================================================================
# Obtain predictions
y_pred = model.predict(data_test)

# Round predictions
y_pred = y_pred.round()
y_pred = [x[0] for x in y_pred]
y_test = list(y_test.values)

# Evaluate results
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(cm)
print("Precision: ", np.round(precision_score(y_test, y_pred, average='macro'), 4))
print("Recall: ", np.round(recall_score(y_test, y_pred, average='macro'), 4))
print("f1_score: ", np.round(f1_score(y_test, y_pred, average='macro'), 4))


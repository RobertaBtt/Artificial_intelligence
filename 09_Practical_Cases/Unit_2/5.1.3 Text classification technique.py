
import pandas as pd
import string

import spacy
import en_core_web_sm
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# load tsv file
df_amazon = pd.read_csv('00-Support Material/datasets/win/amazon_alexa.csv', sep=',')

#top 5 records
print(df_amazon.head())

# shape of datafrme
print(df_amazon.shape)

#view data info
print(df_amazon.info())

# feedback value count
print(df_amazon.feedback.value_counts())


# SPlit text into tokens
# some preprocessing like make all text lowercase, remove stopwords and others

# crate list of punctuation marks
punctuations = string.punctuation

#create list of stopwords
nlp = spacy.load('en_core_web_sm')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

#load english tokenizer, tagger, parser, NER and word vectors
parser = English()

#create tokenizer function
def spacy_tokenizer(sentence):
    #create token object 
    mytokens = parser(sentence)
    
    #lemmatize each token and convert to lowercase
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    
    #remove stop words
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations ]
    
    return mytokens

#Custom transformer using spacy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        #clean text
        return[clean_text(text) for text in X]
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def get_params(self, deep=True):
        return {}

def clean_text(text):
    #remove spaces and convert text to lowercase
    return text.strip().lower()


# when text is classified, we end up with text fragments that match their labels.
#  we cannot use text strings in ml model
#  we need a way to convert text to something that can be represented numerically (1 for positive and 0 for negatvie)
#  classifying text with positive and negative labels is called sentiment analysis.
#  BoW converts text to word occurrence matrix within a given document.
#  it focuses on whether or not th egiven words occurred in the document and genrates a vector that
#  we could see referred to as a BoW vector or a document-term matrix.

bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))


# We also want to look at the TF-IDF (term frequency inverse document frequency) for the terms
#  this is a normalization of our Bag of Words by looking at the frequency of each word compared to the frequency in the document.
#  it is a way to represent the importance of a term in the context of a document based on how many times it appears
# and in how many other documents the same term appears.
#  higher TF-IDF, the more important the term

tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)

# We are trying to build a calssification model, but we need a way to know how it is performing

# Train Test Split

X = df_amazon['verified_reviews']  #features to analyze
ylabels = df_amazon['feedback']  #answers we want to test against

X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)

# CREATE MODEL, TRAIN IT AND MAKE PREDICTIONS

# Logistic Regression Classifier
classifier = LogisticRegression()

# Create pipeline using Bag of Words
pipe = Pipeline([('cleaner', predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', classifier)])

pipe.fit(X_train, y_train)

# predict with test dataset
predicted = pipe.predict(X_test)

#evaluate model
# MOdel Accuracy
print('Logistic Regression Accuracy:', metrics.accuracy_score(y_test, predicted))
print('Logistic Regression Precision:', metrics.precision_score(y_test, predicted))
print('Logistic Regression Recall:', metrics.recall_score(y_test, predicted))


import pandas as pd
import string

import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Load tsv file
df_amazon = pd.read_csv('00-Support Material/datasets/win/amazon_alexa.tsv', sep='\t')

# Top 5 records
print(df_amazon.head())

# Shape of dataframe
print(df_amazon.shape)

# View data info
print(df_amazon.info())

# Feedback value count
print(df_amazon.feedback.value_counts())

# Split text into tokens
# Some preprocessing like making all text lowercase, removing stopwords, and others

# Create list of punctuation marks
punctuations = string.punctuation

# Create list of stopwords
nlp = spacy.load('en_core_web_sm')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER, and word vectors
parser = English()

# Create tokenizer function
def spacy_tokenizer(sentence):
    # Process the sentence using the nlp object
    doc = nlp(sentence)
    
    # Lemmatize each token and convert to lowercase
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in doc]
    
    # Only keep words that are not stop words and not punctuation
    mytokens = [word for word in mytokens if word.isalpha() and word not in stop_words]
    
    return mytokens



# Custom transformer using spacy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Clean text
        return [clean_text(text) for text in X]
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def get_params(self, deep=True):
        return {}

def clean_text(text):
    # Remove spaces and convert text to lowercase
    return text.strip().lower()

# When text is classified, we end up with text fragments that match their labels.
# We cannot use text strings in an ML model
# We need a way to convert text to something that can be represented numerically (1 for positive and 0 for negative)
# Classifying text with positive and negative labels is called sentiment analysis.
# BoW converts text to a word occurrence matrix within a given document.
# It focuses on whether or not the given words occurred in the document and generates a vector that
# we could see referred to as a BoW vector or a document-term matrix.

bow_vector = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 1))

# We also want to look at the TF-IDF (term frequency-inverse document frequency) for the terms
# This is a normalization of our Bag of Words by looking at the frequency of each word compared to the frequency in the document.
# It is a way to represent the importance of a term in the context of a document based on how many times it appears
# and in how many other documents the same term appears.
# Higher TF-IDF, the more important the term

tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer)

# Train Test Split
X = df_amazon['verified_reviews']  # Features to analyze
ylabels = df_amazon['feedback']  # Answers we want to test against

X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3, random_state=42)

# Check if there are some words in the processed documents
sample_texts = X_train.sample(5)  # Take a small sample of the training data
for text in sample_texts:
    print(f"Original Text: {text}")
    print(f"Processed Tokens: {spacy_tokenizer(text)}")
    print()

# Logistic Regression Classifier
classifier = LogisticRegression()

# Create pipeline using Bag of Words
pipe = Pipeline([('cleaner', predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', classifier)])

# Fit the model
pipe.fit(X_train, y_train)

# Predict with the test dataset
predicted = pipe.predict(X_test)

# Evaluate model
# Model Accuracy
print('Logistic Regression Accuracy:', metrics.accuracy_score(y_test, predicted))
print('Logistic Regression Precision:', metrics.precision_score(y_test, predicted))
print('Logistic Regression Recall:', metrics.recall_score(y_test, predicted))

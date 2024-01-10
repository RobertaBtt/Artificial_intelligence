import nltk
#nltk.download('punkt')         #download to run html.parser in BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from bs4 import BeautifulSoup
import urllib.request 

#get text from the web

response = urllib.request.urlopen('https://crayonsforfel.blogspot.com/2023/09/truth-at-1-am.html')
html = response.read()
soup = BeautifulSoup(html, 'html.parser')
text = soup.get_text(strip=True)

characters = ['~','`','!','@','#','$','%','^','&','*','(',')','-','_','+','=',';',':','\'','"','/','?','.',',','<','>','{','}','[',']','\\','|']

# tokenize text.  Split text into individual words
tokens = word_tokenize(text)

# statistical analysis.  see how often certain words appear and show the first 20 most frequent words

freq = nltk.FreqDist(tokens)

for key,val in freq.items():
    print(str(key) + ':' + str(val))

freq.plot(20, cumulative=False)

#  Preprocess text by eliminating stopwords

stopwords.words('english')
clean_tokens = tokens[:]
sr = stopwords.words('english')

for token in tokens:
    if token in stopwords.words('english'):
        clean_tokens.remove(token)
    if token in characters:
        clean_tokens.remove(token)


# Recalculate frequencies and show results
#irrelevant words have been removed so we have a better idea of relevance in text

freq = nltk.FreqDist(clean_tokens)

for key,val in freq.items():
    print(str(key) + ':' + str(val))

freq.plot(20, cumulative=False)

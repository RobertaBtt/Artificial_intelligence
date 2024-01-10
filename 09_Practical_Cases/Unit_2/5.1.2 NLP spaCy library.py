import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

spacy.load('en_core_web_sm')
import en_core_web_sm


from spacy import displacy



#tokenization

#load english tokenizer, tagger, parser, NER and word vectors
nlp = English()

text = """When learning data science, you shouldn't get discouraged!  Challenges and setbacks aren't failures, they are just part of the journey.  You've got this!"""

# nlp object is used to create documents with linguistic annotations
my_doc = nlp(text)

#create list of word tokens
token_list = []
for token in my_doc:
    token_list.append(token.text)

print(token_list)


# Sentence tokenization

#load english tokenizer, tagger, parser, NER and word vectors
nlp = English()

# add pipleline 'sentencizer' component
sbd = nlp.add_pipe('sentencizer')



text = """When learning data science, you shouldn't get discouraged!  Challenges and setbacks aren't failures, they are just part of the journey.  You've got this!"""

# nlp object is used to create documents with linguistic annotations
doc = nlp(text)

#create list of word tokens
sentence_list = []
for sent in doc.sents:
    sentence_list.append(sent.text)

print(sentence_list)


# Stop words
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

#print total number of stop words:
print('number of stop words: %d' % len(spacy_stopwords))

#print first ten stop words:
print('first ten stop words: %s' % list(spacy_stopwords)[:20])

# implementation of stopwords:
    
filtered_sent = []

# nlp object used to create documents with linguistic annotations.
doc = nlp(text)

#filter stop words
for word in doc:
    if word.is_stop==False:
        filtered_sent.append(word)

print('filtered sentence:', filtered_sent)

# first get the number of words which are considered to be of this type in english
# a total of 326, of which the first 20 are those that we show as first ten stop words
# eliminate these types of results from text and show words that are actually valid

print('number of stop words:  ', len(STOP_WORDS))

print('first twenty stopwords:', list(STOP_WORDS)[:20])


# word tagging

# load en_core_web_sm of english for vocabulary, sytax and entities
nlp = en_core_web_sm.load()

# nlp object used to crate docs
docs = nlp(u'All is well that ends well.')

for word in docs:
    print(word.text,word.pos_)
    
# ENTITY DETECTION
#  Also called entity recognition is advanced for m of lanugage processing
#  it identifies important elements like places, people, organizations and languages
#  within a text string.
#  useful for quickly extracting info from text
#  since it is possible to select important topics or identify key sections of text


nytimes = nlp(u"""New York City on Tuesday declared a public health emergency and
              ordered mandatory measles vaccinations amid an outbreak, becoming the
              latest national flash point over refusals to inoculate against dangerous diseases.
              
              At least 285 people have contracted measles in the city since September, 
              mostly in Brooklyn's Williamsburg neighborhood.  The order covers four Zip
              codes there, Mayor Bill de Blasio (D) said Tuesday.
              
              The mandate orders all unvaccinated people in the area, including
              a concentration of Orthodox Jews, to receive inoculations, including 
              for children as young as 6 months old.  Anyone who resists could be
              fined up to $1,000.""")
              
entities = [(i, i.label_, i.label) for i in nytimes.ents]
entities

displacy.render(nytimes, style='ent', jupyter=True)


# DEPENDENCY ANALYSIS
#  Language processing technique that lets us better determine the
#  meaning of a sentence by analyzing how it is constructed to determine how
#  individual words are related to one another

docp = nlp('In pursuit of a wall, President Trump ran into one.')

for chunk in docp.noun_chunks:
    print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)
    
displacy.render(docp, style='dep', jupyter=True)



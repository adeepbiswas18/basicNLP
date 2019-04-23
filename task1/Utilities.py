import nltk, logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.neural_network import MLPClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def getContents(type):
    contents = []
    labels1 = []

    #loading the training data
    if type == 'train':
        twefile = "intenttrain.csv"
        data = []
        df = pd.read_csv(twefile)
        tweets = df.iloc[:, 1]
        cats = df.iloc[:, 0]
        for row in range(0, tweets.shape[0]):
            tweet = tweets[row]
            contents.append(tweet)
        for row in range(0, cats.shape[0]):
            cat = cats[row]
            labels1.append(cat)
        labels = np.asarray(labels1)

    #loading the testing data
    elif type == 'test':
        twefile = "intenttest.csv"
        data = []
        df = pd.read_csv(twefile)
        tweets = df.iloc[:, 1]
        cats = df.iloc[:, 0]
        for row in range(0, tweets.shape[0]):
            tweet = tweets[row]
            contents.append(tweet)
        for row in range(0, cats.shape[0]):
            cat = cats[row]
            labels1.append(cat)
        labels = np.asarray(labels1)

    return {'Contents':contents, 'Labels':labels}

def myTokenizer(text):
    #cleaning the text data before tockenizing
    return nltk.regexp_tokenize(text, "\\b[a-zA-Z]{3,}\\b")

def tokenizeContents(contents):
    #returns the tocken contents
    return [myTokenizer(content) for content in contents]

#vectorize the tockens for mapping
def getVectorizer():
    vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', tokenizer=myTokenizer)

    return vectorizer

#breaks the text into ngrams
def breakWordIntoNgrams(word, ngram_size=2):
    ngrams = []
    for i in range(len(word)-ngram_size+1):
        ngrams.append(word[i:i + ngram_size])

    return ngrams

#generate input and ouput for training mlp models
def getNNTrainableInstances(contents, min_ngram_size=1, max_ngram_size=3):
    vectorizer = getVectorizer()
    vectorizer.fit_transform(contents)
    vocabulary = set(vectorizer.vocabulary_.keys())
    docWords = tokenizeContents(contents)
    inputs = []
    outputs = []
    for words in docWords:
        words = [word.lower() for word in words]
        wordSet = set.intersection(set(words), vocabulary)
        for word in wordSet:
            tokens = []
            for ngram_size in range(min_ngram_size, max_ngram_size + 1):
                tokens = tokens + breakWordIntoNgrams(word, ngram_size)
            for i in range(len(tokens)):
                out = tokens[i]
                inp = tokens[0:i] + tokens[i+1:len(tokens)]
                inputs.append(" ".join(inp))
                outputs.append(out)
    return dict({'Inputs':inputs, 'Outputs':outputs})

#calling functions for tesing
contents = getContents('train')
instances = getNNTrainableInstances(contents['Contents'][0:1000], max_ngram_size=2)
vectorizer = CountVectorizer(binary=True, token_pattern='\\b[a-zA-Z]+\\b')
trainData = vectorizer.fit_transform(instances['Inputs'])
trainLabels = instances['Outputs']

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100), random_state=1)
clf.fit(trainData, trainLabels)



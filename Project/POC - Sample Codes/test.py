
from models.Model_LSTM_1 import LSTMModel

print("Hello world")

import pathlib
print(pathlib.Path("__file__").parent.absolute())
import os

path = os.path.dirname(__file__)

os.chdir(path)

print(os.path.dirname(__file__))

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ['bromwell high cartoon comedy']

y = [1]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.shape)
classifier = MultinomialNB()
classifier.fit(X, y)

print(type(X),type(y))
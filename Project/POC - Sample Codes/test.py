
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

review_ids = []
for i in range(1,6):
    review_ids.append(randint(1,4501))
    
print(review_ids)
from operations import *
with app.app_context():
    reviewss = get_reviews(review_ids)
print(reviewss)


from random import seed, randint
seed(1)
print(randint(1,100))
seed(2)

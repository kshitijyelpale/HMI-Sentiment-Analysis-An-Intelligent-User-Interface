from lstm_model import LSTMModel

x_train = []
y_train = [0]*22750 + [1]*22750
x_test = []
y_test = [0]*2250 + [1]*2250
lstmModel = LSTMModel()
x_train, x_test = lstmModel.importData(x_train, x_test)

#############################################

from sklearn.metrics import confusion_matrix

#############################################
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
#stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf = True, lowercase = True, strip_accents = 'ascii')

x_train, x_test = lstmModel.preProcessData(x_train, x_test)

X_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.transform(x_test)


#############################################
from sklearn.linear_model.logistic import LogisticRegression
model = LogisticRegression(max_iter = 10000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("Accuracy is " + str(((cm[0,0] + cm[1,1]) / len(y_test)) * 100))


#############################################
from sklearn.naive_bayes import MultinomialNB
model2 = MultinomialNB()
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)

cm2 = confusion_matrix(y_test, y_pred2)

print(cm2)

print("Accuracy is " + str(((cm2[0,0] + cm2[1,1]) / len(y_test)) * 100))

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, model2.predict_proba(X_test)[:,1])

import numpy as np
movie_reviews_array = np.array(["Jupiter Ascending was a disappointing and terrible movie"])

movie_review_vector = vectorizer.transform(movie_reviews_array)

print(model2.predict(movie_review_vector))


from sklearn.naive_bayes import GaussianNB
model22 = GaussianNB()
model22.fit(X_train, y_train)
y_pred22 = model22.predict(X_test)

cm22 = confusion_matrix(y_test, y_pred22)

print(cm22)

print("Accuracy is " + str(((cm22[0,0] + cm22[1,1]) / len(y_test)) * 100))

roc_auc_score(y_test, model22.predict_proba(X_test)[:,1])

#############################################
from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier(max_depth=2, random_state=0)
model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)

cm3 = confusion_matrix(y_test, y_pred3)

print(cm3)

print("Accuracy is " + str(((cm3[0,0] + cm3[1,1]) / len(y_test)) * 100))

#############################################
from sklearn.tree import DecisionTreeClassifier
model4 = DecisionTreeClassifier(random_state=0)
model4.fit(X_train, y_train)
y_pred4 = model4.predict(X_test)

cm4 = confusion_matrix(y_test, y_pred4)

print(cm4)

print("Accuracy is " + str(((cm4[0,0] + cm4[1,1]) / len(y_test)) * 100))

#############################################
from sklearn.neighbors import KNeighborsClassifier
model5 = KNeighborsClassifier(n_neighbors=5)
model5.fit(X_train, y_train)
y_pred5 = model5.predict(X_test)

cm5 = confusion_matrix(y_test, y_pred5)

print(cm5)

print("Accuracy is " + str(((cm5[0,0] + cm5[1,1]) / len(y_test)) * 100))


#############################################
from sklearn.svm import SVC
model6 = SVC()
model6.fit(X_train, y_train)
y_pred6 = model6.predict(X_test)

cm6 = confusion_matrix(y_test, y_pred6)

print(cm6)

print("Accuracy is " + str(((cm6[0,0] + cm6[1,1]) / len(y_test)) * 100))

from abc import ABC, abstractmethod

from datapreprocessing import *


class MLModelTemplate(ABC):
    def __init__(self):
        pass

    # Method for pre-processing data
    def preProcessData(self, x_train, x_test):
        from datapreprocessing import remove_stopwords_and_special_chars

        print("Pre-processing data... Please wait!")

        # Fetch all stopwords and keep required stopwords
        x_train = remove_stopwords_and_special_chars(x_train)
        x_test = remove_stopwords_and_special_chars(x_test)

        print("Pre-processing done...!!!")
        return x_train, x_test

    @abstractmethod
    def create_model(self):
        pass

    def train_model(self, model, X_train, y_train):
        model.fit(X_train, y_train)

        return model

    def predict(self, model, data):
        return model.predict(data)

    def get_accuracy(self, y_true, y_pred):
        from sklearn.metrics import accuracy_score

        # roc_auc_score(y_test, model22.predict_proba(X_test)[:,1])

        return accuracy_score(y_true, y_pred)

    def get_confusion_matrix(self, y_true, y_pred):
        from sklearn.metrics import confusion_matrix

        return confusion_matrix(y_true, y_pred)

    def execute(self):
        x_train = []
        y_train = [0] * 22750 + [1] * 22750
        x_test = []
        y_test = [0] * 2250 + [1] * 2250

        x_train, x_test = import_data(x_train, x_test)

        x_train, x_test = self.preProcessData(x_train, x_test)

        X_train, X_test = tfidfvectorizer(x_train, x_test)

        model = self.create_model()

        model = self.train_model(model, X_train, y_train)

        y_pred = self.predict(model, X_test)

        cm = self.get_confusion_matrix(y_test, y_pred)

        print(cm)

        print("Accuracy is " + str(self.get_accuracy(y_test, y_pred) * 100))

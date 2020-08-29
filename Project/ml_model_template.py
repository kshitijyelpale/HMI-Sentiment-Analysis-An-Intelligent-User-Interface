from abc import ABC, abstractmethod
import os

from ml_utilities import save_ml_model, load_ml_model


class MLModelTemplate(ABC):
    def __init__(self):
        pass

    # Method for pre-processing data
    def pre_process_data(self, x_train, x_test):
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
        return model.predict_proba(data)

    def get_accuracy(self, y_true, y_pred):
        from sklearn.metrics import accuracy_score

        # roc_auc_score(y_test, model22.predict_proba(X_test)[:,1])

        return accuracy_score(y_true, y_pred)

    def get_confusion_matrix(self, y_true, y_pred):
        from sklearn.metrics import confusion_matrix

        return confusion_matrix(y_true, y_pred)

    def predict_reviews(self):
        pass

    def save_model(self, model, ext=''):
        path = os.path.dirname(__file__)
        name = type(self).__name__
        if ext:
            name += "_" + ext
        save_ml_model(model, path, name)
        print("Model saved...")

    def load_model(self, ext=''):
        path = os.path.dirname(__file__)
        name = type(self).__name__
        if ext:
            name += "_" + ext
        return load_ml_model(path, name)

    def execute(self):
        from datapreprocessing import import_data, tfidfvectorizer

        x_train, y_train, x_test, y_test = import_data(False)
        
        x_train, x_test = self.pre_process_data(x_train, x_test)
        
        vec, x_train, x_test = tfidfvectorizer(x_train, x_test)
        
        self.save_model(vec, 'vec')

        model = self.create_model()

        # model = self.train_model(model, x_train, y_train)

        model.fit(x_train, y_train)

        self.save_model(model)

        model = self.load_model()

        y_pred = self.predict(model, x_test)

        # cm = self.get_confusion_matrix(y_test, y_pred)

        # print(cm)

        # print("Accuracy is " + str(self.get_accuracy(y_test, y_pred) * 100))

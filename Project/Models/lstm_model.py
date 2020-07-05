# import libraries
import os
import re
import sys
import nltk

try:
    from nltk.corpus import stopwords
except:
    nltk.download('stopwords')
    from nltk.corpus import stopwords

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM

from utils.utilities import *

from datapreprocessing import *


# define class
class LSTMModel:

    def __init__(self):
        pass

    # Method for pre-processing data
    def preProcessData(self, x_train, x_test):
        print("Pre-processing data... Please wait!")

        # Fetch all stopwords and keep required stopwords
        x_train = remove_stopwords_and_special_chars(x_train)
        x_test = remove_stopwords_and_special_chars(x_test)

        print("Pre-processing done...!!!")
        return x_train, x_test

    # Method for Vectorizing(One-Hot) and Encoding data
    def encodeData(self, max_features, max_doc_len, x_train, x_test):
        print("Encoding data to One Hot Representation...")

        x_train = onehot_encoding(x_train, max_features, max_doc_len)
        x_test = onehot_encoding(x_test, max_features, max_doc_len)

        print("Encoding done...!!!")
        return x_train, x_test

    # Method for creating ML->RNN->LSTM model
    def createModel(self, max_features):
        print("Creating model...")
        model = Sequential()
        # Layer 1-> Embedding
        model.add(Embedding(max_features, 128))
        # Layer 2-> LSTM
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        # Layer 3-> Fully Connected (Dense)
        model.add(Dense(1, activation='sigmoid'))
        # Choose best optimizer
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("Model created...!!!")
        return model

    # Method for training the model
    def trainModel(self, model, x_train, y_train, x_test, y_test):
        print("Training Model... Please wait!")
        model.fit(x_train, y_train, batch_size=32, epochs=15, validation_data=(x_test, y_test))
        print("Model trained...!!!")
        return model

    # Method for evaluating the model using test-data
    def validateModel(self, model, x_test, y_test):
        score, acc = model.evaluate(x_test, y_test, batch_size=32)
        print('Test score:', score)
        print('Test accuracy:', acc)

    # Method for predicting results through trained model
    def predict(self, model, x_test):
        output = model.predict(x_test)
        # print(output.shape)
        # print(output)

        return output

    # Method for saving trained model
    def saveModel(self, model):
        path = os.path.dirname(__file__)

        save_nn_model(model, path, "lstm_model")
        print("Model saved...")

    # Method for loading saved model
    def loadModel(self):
        filename = path = os.path.dirname(__file__) + "/lstm_model"

        loaded_model = load_nn_model(filename)
        print("Model loaded successfully...")

        return loaded_model


def main():
    try:
        # Declare train and test Variables
        x_train = []
        y_train = [0] * 22750 + [1] * 22750
        x_test = []
        y_test = [0] * 2250 + [1] * 2250
        # Define vocabulary size
        max_features = 47000
        # Define number of words per document/review
        max_doc_len = 220

        lstmModel = LSTMModel()
        x_train, x_test = import_data(x_train, x_test)
        x_train, x_test = lstmModel.preProcessData(x_train, x_test)
        x_train, x_test = lstmModel.encodeData(max_features, max_doc_len, x_train, x_test)
        model = lstmModel.createModel(max_features)
        model = lstmModel.trainModel(model, x_train, y_train, x_test, y_test)
        lstmModel.validateModel(model, x_test, y_test)
        lstmModel.predict(model, x_test)

        # Save the trained model
        # lstmModel.saveModel(model)

        # Load the saved model
        model = lstmModel.loadModel()

        # Try with new review
        new_review = ["I am very happy after watching this movie", "I am very sad"]
        temp = []
        temp, new_review = lstmModel.preProcessData(temp, new_review)
        temp, new_review = lstmModel.encodeData(max_features, max_doc_len, temp, new_review)

        lstmModel.predict(model, new_review)

    except:
        print("Unexpected error:", sys.exc_info()[0:2])


if __name__ == "__main__":
    main()

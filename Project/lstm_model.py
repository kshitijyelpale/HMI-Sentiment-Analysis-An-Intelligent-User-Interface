# import libraries
import os
import sys
import nltk

try:
    from nltk.corpus import stopwords
except:
    nltk.download('stopwords')
    from nltk.corpus import stopwords

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM

from ml_utilities import *

from ml_model_template import MLModelTemplate


# define class
class LSTMModel(MLModelTemplate):

    def __init__(self):
        pass

    # Method for Vectorizing(One-Hot) and Encoding data
    def encode_data(self, max_features, max_doc_len, x_train, x_test):
        
        from Models.datapreprocessing import onehot_encoding
        
        print("Encoding data to One Hot Representation...")

        x_train = onehot_encoding(x_train, max_features, max_doc_len)
        x_test = onehot_encoding(x_test, max_features, max_doc_len)

        print("Encoding done...!!!")
        return x_train, x_test

    # Method for creating ML->RNN->LSTM model
    def create_model(self, max_features):
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
    def train_model(self, model, x_train, y_train, x_test, y_test):
        print("Training Model... Please wait!")
        model.fit(x_train, y_train, batch_size=32, epochs=15, validation_data=(x_test, y_test))
        print("Model trained...!!!")
        return model

    # Method for evaluating the model using test-data
    def validate_model(self, model, x_test, y_test):
        score, acc = model.evaluate(x_test, y_test, batch_size=32)
        print('Test score:', score)
        print('Test accuracy:', acc)


    # Method for saving trained model
    def save_model(self, model):
        path = os.path.dirname(__file__)

        save_nn_model(model, path, "lstm_model")
        print("Model saved...")

    # Method for loading saved model
    def load_model(self):
        filename = path = os.path.dirname(__file__) + "/lstm_model"

        loaded_model = load_nn_model(filename)
        print("Model loaded successfully...")

        return loaded_model
        

    def execute(self):
        try:
            # Define vocabulary size
            max_features = 47000
            # Define number of words per document/review
            max_doc_len = 220

            from datapreprocessing import import_data
            x_train, y_train, x_test, y_test = import_data(False)
            x_train, x_test = self.pre_process_data(x_train, x_test)
            x_train, x_test = self.encode_data(max_features, max_doc_len, x_train, x_test)

            model = self.create_model(max_features)
            #model = lstmModel.train_model(model, x_train, y_train, x_test, y_test)
            self.validate_model(model, x_test, y_test)
            self.predict(model, x_test)

            # Save the trained model
            # lstmModel.save_model(model)

            # Load the saved model
            model = self.load_model()

            # Try with new review
            new_review = ["I am very happy after watching this movie", "I am very sad"]
            temp = []
            temp, new_review = self.pre_process_data(temp, new_review)
            temp, new_review = self.encode_data(max_features, max_doc_len, temp, new_review)

            self.predict(model, new_review)

        except:
            print("Unexpected error:", sys.exc_info()[0:2])
            
    def predict_reviews(self, raw_data):
        temp = []
        temp, data = self.pre_process_data(temp, raw_data)
        temp, data = self.encode_data(47000, 220, temp, data)
        model = self.load_model()
        return self.predict(model, data)


if __name__ == "__main__":
    lstm = LSTMModel()
    #lstm.execute()
    
    reviews = ["I am very happy after watching this movie"]
    result = lstm.predict_reviews(reviews)
    print(result[0][0])

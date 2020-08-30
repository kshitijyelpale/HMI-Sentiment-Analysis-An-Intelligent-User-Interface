# import libraries
import os
import sys
from keras.datasets import imdb
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers
from datapreprocessing import remove_stopwords_and_special_chars
from ml_model_template import MLModelTemplate


# define class
class BiLSTMModel(MLModelTemplate):

    def __init__(self):
        self.__max_features = 200000  # Only consider the top 20k words
        self.__maxlen = 200  # Only consider the first 200 words of each movie review

    # Load the IMDB movie review sentiment data
    def load_data(self):
        (self.__x_train, self.__y_train), (self.__x_val, self.__y_val) = keras.datasets.imdb.load_data(
            num_words=self.__max_features)
        self.__x_train = keras.preprocessing.sequence.pad_sequences(self.__x_train, maxlen=self.__maxlen)
        self.__x_val = keras.preprocessing.sequence.pad_sequences(self.__x_val, maxlen=self.__maxlen)

    # Method for creating ML->RNN->Bi_LSTM model
    def create_model(self):
        print("Creating model...")
        # Input for variable-length sequences of integers
        inputs = keras.Input(shape=(None,), dtype="int32")
        # Embed each integer in a 128-dimensional vector
        x = layers.Embedding(self.__max_features, 128)(inputs)
        # Add 2 bidirectional LSTMs
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(64))(x)
        # Add a classifier
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)
        print("Model created...!!!")
        return model

    # Method for training the model
    def train_model(self, model):
        print("Training Model... Please wait!")
        model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
        model.fit(self.__x_train, self.__y_train, batch_size=32, epochs=2, validation_data=(self.__x_val, self.__y_val))
        print("Model trained...!!!")
        return model

    # Method for saving trained model
    def save_model(self, model):
        filename = os.path.dirname(__file__) + "/bi_lstm_weights.h5"
        model.save(filename)
        print("Model saved...")

    # Method for loading saved model
    def load_model(self):
        filename = os.path.dirname(__file__) + "/bi_lstm_weights.h5"
        return load_model(filename)
    
    def predict(self, model, data):
        return model.predict(data)

    def execute(self, new_review):
        try:
            new_review = remove_stopwords_and_special_chars(new_review)
            word_indices = imdb.get_word_index()
            reviews = []
            for doc in new_review:
                review = []
                for word in doc:
                    if word not in word_indices:
                        review.append(2)
                    else:
                        review.append(word_indices[word] + 3)
                review.sort(reverse=True)
                reviews.append(review)
            x_test = keras.preprocessing.sequence.pad_sequences(reviews, truncating='post', padding='post',
                                                                maxlen=self.__maxlen)
            return x_test

        except:
            print("Unexpected error:", sys.exc_info()[0:2])

    def predict_reviews(self, raw_data):
        #data = self.execute(raw_data)
        #self.load_data()
        #model = self.create_model()
        #model = self.train_model(model)
        #return model.predict(data)
    
        model = self.load_model()
        data = self.execute(raw_data)
        return self.predict(model, data)
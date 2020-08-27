# import libraries
import os
import sys
import nltk

try:
    from nltk.corpus import stopwords
except:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
from keras.datasets import imdb
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM

from models.ml_utilities import *

from models.ml_model_template import MLModelTemplate


# define class
class BiLSTMModel(MLModelTemplate):

    def __init__(self):
        self.__max_features = 20000  # Only consider the top 20k words
        self.__maxlen = 200  # Only consider the first 200 words of each movie review

    #Load the IMDB movie review sentiment data
    def load_data(self):
        (self.__x_train, self.__y_train), (self.__x_val, self.__y_val) = keras.datasets.imdb.load_data(num_words=self.__max_features)
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
        path = os.path.dirname(__file__)

        save_nn_model(model, path, "bi_lstm_model")
        print("Model saved...")

    # Method for loading saved model
    def load_model(self):
        filename = path = os.path.dirname(__file__) + "/bi_lstm_model"

        loaded_model = load_nn_model(filename)
        print("Model loaded successfully...")

        return loaded_model
        

    def execute(self):
        try:
            # Load the saved model
            model = self.load_model()

            # Try with new review
            new_review = ["""I absolutely adored this movie. For me, the best reason to see it is how stark a contrast it is from legal dramas like "Boston Legal" or "Ally McBeal" or even "LA Law." This is REALITY. The law is not BS, won in some closing argument or through some ridiculous defense you pull out of your butt, like the "Chewbacca defense" on South Park.) This is a real travesty of justice, the legal system gone horribly wrong, and the work by GOOD lawyers - not the shyster stereotype, who use all of their skills to right it. It will do more for restoring your faith in humanity than any Frank Capra movie or TO KILL A MOCKINGBIRD. And most importantly, I wept. During the film, during the featurette included at the end of the DVD - it's amazing. Wonderful film; wonderfully made. Thank God the filmmakers made it."""]
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
            x_test = keras.preprocessing.sequence.pad_sequences(reviews, truncating = 'post', padding = 'post', maxlen = self.__maxlen)
            #print(model.predict(x_test))

            self.predict(model, new_review)

        except:
            print("Unexpected error:", sys.exc_info()[0:2])
            
    def predict_reviews(self, raw_data):
        model = self.load_model()
        return self.predict(model, data)


if __name__ == "__main__":
    bi_lstm = BiLSTMModel()
    #bi_lstm.execute()
    
    reviews = ["""I absolutely adored this movie. For me, the best reason to see it is how stark a contrast it is from legal dramas like "Boston Legal" or "Ally McBeal" or even "LA Law." This is REALITY. The law is not BS, won in some closing argument or through some ridiculous defense you pull out of your butt, like the "Chewbacca defense" on South Park.) This is a real travesty of justice, the legal system gone horribly wrong, and the work by GOOD lawyers - not the shyster stereotype, who use all of their skills to right it. It will do more for restoring your faith in humanity than any Frank Capra movie or TO KILL A MOCKINGBIRD. And most importantly, I wept. During the film, during the featurette included at the end of the DVD - it's amazing. Wonderful film; wonderfully made. Thank God the filmmakers made it."""]
    result = bi_lstm.predict_reviews(reviews)
    print(result[0][0])

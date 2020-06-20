# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 22:45:21 2020

@author: Safir Mohammad
"""

#import libraries
import os
import re
import sys
import nltk
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Embedding
from keras.layers import LSTM
try:
    from nltk.corpus import stopwords
except:
    nltk.download('stopwords')
    from nltk.corpus import stopwords


#define class
class LSTMModel:  
    
    def __init__(self):
        pass
    
    
    #Method for importing dataset
    def importData(self, x_train, x_test):
        
        print("Importing Dataset... Please wait!")
        
        #Import testing data in x_test
        for filename in os.listdir("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\DataSet\\Test\\Negative\\"):
            with open(os.path.join("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\DataSet\\Test\\Negative\\", filename), 'r', encoding="utf8") as f:
                x_test.append(f.read())
      
        for filename in os.listdir("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\DataSet\\Test\\Positive\\"):
            with open(os.path.join("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\DataSet\\Test\\Positive\\", filename), 'r', encoding="utf8") as f:
                x_test.append(f.read())
      
        #Import training data in x_train
        for filename in os.listdir("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\DataSet\\Train\\Negative\\"):
            with open(os.path.join("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\DataSet\\Train\\Negative\\", filename), 'r', encoding="utf8") as f:
                x_train.append(f.read())
      
        for filename in os.listdir("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\DataSet\\Train\\Positive\\"):
            with open(os.path.join("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\DataSet\\Train\\Positive\\", filename), 'r', encoding="utf8") as f:
                x_train.append(f.read())

        print("Dataset successfully imported...!!!")
        return x_train, x_test
    
    
    #Method for pre-processing data
    def preProcessData(self, x_train, x_test):
        
        print("Pre-processing data... Please wait!")
        
        #Fetch all stopwords and keep required stopwords
        all_stopwords = stopwords.words('english')
        my_stopwords = [ word for word in all_stopwords if word not in ("against", "up", "down", "out", "off", "over", "under", "more", "most", "each", "few", "some", "such", "no", "nor", "not", "only", "too", "very", "don", "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't")]
        
        #Get rid of special characters
        REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
        REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
        
        for i in range(0,len(x_test)):
            #Keep only alphabets with single whitespace
            x_test[i] = REPLACE_NO_SPACE.sub("", x_test[i].lower())
            x_test[i] = REPLACE_WITH_SPACE.sub(" ", x_test[i])
            
            #Remove unwanted stopwords
            x_test[i] = x_test[i].split()
            x_test[i] = [ word for word in x_test[i] if word not in my_stopwords]
            x_test[i] = " ".join(x_test[i])
            
        for i in range(0,len(x_train)):
            #Keep only alphabets with single whitespace
            x_train[i] = REPLACE_NO_SPACE.sub("", x_train[i].lower())
            x_train[i] = REPLACE_WITH_SPACE.sub(" ", x_train[i])
            
            #Remove unwanted stopwords
            x_train[i] = x_train[i].split()
            x_train[i] = [ word for word in x_train[i] if word not in my_stopwords]
            x_train[i] = " ".join(x_train[i])
        
        
        print("Pre-processing done...!!!")
        return x_train, x_test
        
    
    #Method for Vectorizing(One-Hot) and Encoding data
    def encodeData(self, max_features, max_doc_len, x_train, x_test):
        
        print("Encoding data to One Hot Representation...")
        
        x_test = [ one_hot(document, max_features) for document in x_test]
        x_train = [ one_hot(document, max_features) for document in x_train]
        
        #Add Bias
        for i in range(0, len(x_test)):
            x_test[i] = [1] + x_test[i]
        for i in range(0, len(x_train)):
            x_train[i] = [1] + x_train[i]
        
        #Word Embedding
        x_test = pad_sequences(x_test, truncating = 'post', padding = 'post', maxlen = max_doc_len)
        x_train = pad_sequences(x_train, truncating = 'post', padding = 'post', maxlen = max_doc_len)
        
        print("Encoding done...!!!")
        return x_train, x_test
    
    
    #Method for creating ML->RNN->LSTM model
    def createModel(self, max_features):
        
        print("Creating model...")
        model = Sequential()
        #Layer 1-> Embedding
        model.add(Embedding(max_features, 128))
        #Layer 2-> LSTM
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        #Layer 3-> Fully Connected (Dense)
        model.add(Dense(1, activation='sigmoid'))
        #Choose best optimizer
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("Model created...!!!")
        return model
    
    
    #Method for training the model
    def trainModel(self, model, x_train, y_train, x_test, y_test):
        
        print("Training Model... Please wait!")
        model.fit(x_train, y_train, batch_size=32, epochs=15, validation_data=(x_test, y_test))
        print("Model trained...!!!")
        return model
    
    
    #Method for evaluating the model using test-data
    def validateModel(self, model, x_test, y_test):
        score, acc = model.evaluate(x_test, y_test, batch_size=32)
        print('Test score:', score)
        print('Test accuracy:', acc)
    
    
    #Method for predicting results through trained model
    def testModel(self, model, x_test):
        output = model.predict(x_test)
        print(output.shape)
        print(output)
        
        
    #Method for saving trained model
    def saveModel(self, model):
        #Serialize to JSON
        json_file = model.to_json()
        with open("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\Model_LSTM.json", "w") as file:
            file.write(json_file)
        
        #Serialize weights to HDF5
        model.save_weights("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\lstm_model_weights.h5")
        print("Model saved...")
        
    
    #Method for loading saved model
    def loadModel(self):
        #Load JSON and create model
        file = open("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\Model_LSTM.json", "r")
        model_json = file.read()
        file.close()
        
        loaded_model = model_from_json(model_json)
        #Load weights
        loaded_model.load_weights("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\lstm_model_weights.h5")
        print("Model loaded successfully...")
        
        return loaded_model
    
    
def main():
    try:
        #Declare train and test Variables
        x_train = []
        y_train = [0]*22750 + [1]*22750
        x_test = []
        y_test = [0]*2250 + [1]*2250
        #Define vocabulary size
        max_features = 47000
        #Define number of words per document/review
        max_doc_len = 220
        
        lstmModel = LSTMModel()
        x_train, x_test = lstmModel.importData(x_train, x_test)    
        x_train, x_test = lstmModel.preProcessData(x_train, x_test)
        x_train, x_test = lstmModel.encodeData(max_features, max_doc_len, x_train, x_test)
        model = lstmModel.createModel(max_features)
        model = lstmModel.trainModel(model, x_train, y_train, x_test, y_test)
        lstmModel.validateModel(model, x_test, y_test)
        lstmModel.testModel(model, x_test)
        
        #Save the trained model
        lstmModel.saveModel(model)
        
        #Load the saved model
        model = lstmModel.loadModel()
        
        #Try with new review
        new_review = ["I am very happy after watching this movie","I am very sad"]
        temp = []
        temp, new_review = lstmModel.preProcessData(temp, new_review)
        temp, new_review = lstmModel.encodeData(max_features, max_doc_len, temp, new_review)
        
        lstmModel.testModel(model, new_review)
        
    except:
        print("Unexpected error:", sys.exc_info()[0:2])
    

if __name__ == "__main__":
    main()
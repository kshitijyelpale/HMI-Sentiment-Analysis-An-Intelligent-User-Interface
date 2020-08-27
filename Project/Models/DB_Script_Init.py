# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:09:09 2020

@author: Safir Mohammad
"""

#import libraries
import os
import sys
sys.path.append('../')
from Models.bi_lstm_model import BiLSTMModel
from Models.naive_bayes_model import NaiveBayesModel
from dbops.models import Reviews, db

#define class
class DBInitLoad:  
    
    def __init__(self):
        self.__rev_lst, self.__rat_lst = [], []
        self.__lstm_rat, self.__bayes_rat = [], []
        self.__prim_key = list(range(4500))

    #Method for importing dataset
    def importData(self):
        #Import data
        for filename in os.listdir("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\DataSet\\Test\\Negative\\"):
            with open(os.path.join("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\DataSet\\Test\\Negative\\", filename), 'r', encoding="utf8") as f:
                st_ind = list(filename).index("_")
                nd_ind = list(filename).index(".")
                self.__rev_lst.append(f.read())
                self.__rat_lst.append(int(filename[st_ind+1:nd_ind])/10)
      
        for filename in os.listdir("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\DataSet\\Test\\Positive\\"):
            with open(os.path.join("F:\\Master's\\SS20\\HMI\\SentimentAnalysis\\DataSet\\Test\\Positive\\", filename), 'r', encoding="utf8") as f:
                st_ind = list(filename).index("_")
                nd_ind = list(filename).index(".")
                self.__rev_lst.append(f.read())
                self.__rat_lst.append(int(filename[st_ind+1:nd_ind])/10)
      
    def predictAll(self):
        self.__lstm_rat = BiLSTMModel().predict_reviews(self.__rev_lst)
        self.__lstm_rat = [j for sub in self.__lstm_rat for j in sub]
        self.__bayes_rat = NaiveBayesModel().predict_reviews(self.__rev_lst)
        self.__bayes_rat = [j for sub in self.__bayes_rat for j in sub] 

        
    def updateDB(self):
        for i in range(4500):
            r = Reviews(self.__rev_lst[i], self.__lstm_rat[i], self.__bayes_rat[i], self.__rat_lst[i])
            db.session.add(r)
        db.session.commit()
    
    def printAll(self):
        print(self.__lstm_rat, self.__bayes_rat)
        
    
def main():
    print("Loading Data... Please wait!")
    db_obj = DBInitLoad()    
    db_obj.importData()
    db_obj.predictAll()
    db_obj.updateDB()
    db_obj.printAll()

if __name__ == "__main__":
    main()
# import libraries
import sys
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
sys.path.append('../')
# from dbops.models import Reviews, db
from operations import *

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///db.sentiment_analysis1'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)


# define class
class CorrelationCoeff:

    def __init__(self):
        self.__rat_lst = [], []
        self.__lstm_rat, self.__bayes_rat = [], []
        
    def getter(self):
        return self.__rat_lst,  self.__lstm_rat, self.__bayes_rat

    # Method for importing dataset
    def importData(self):
        # Import data
        self.__rat_lst = get_actual_sentiments()
        self.__lstm_rat = get_lstm_predictions()
        self.__bayes_rat = get_bayes_predictions()

    def perform_correlation(self):
        lstm_r, _ = pearsonr(self.__rat_lst, self.__lstm_rat)
        bayes_r, _ = pearsonr(self.__rat_lst, self.__bayes_rat)
        return lstm_r, bayes_r
    
    def spearman_correlation(self):
        lstm_r, _ = spearmanr(self.__rat_lst, self.__lstm_rat)
        bayes_r, _ = spearmanr(self.__rat_lst, self.__bayes_rat)
        return lstm_r, bayes_r

    def printAll(self):
        print(len(self.__lstm_rat), len(self.__bayes_rat))
        
    def ttest(self):
        from scipy import stats
        a,b = get_deviations()
        t, p = stats.ttest_ind(a,b)
        
        print(t, p)
        
        import matplotlib.pyplot as plt
        fig1, ax1 = plt.subplots()
        #ax1.set_title('Lstm Deviations')
        #ax1.boxplot(a)
        
        #fig1, ax1 = plt.subplots()
        #ax1.set_title('bayes Deviations')
        #ax1.boxplot(b)
        
        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as stats
        import math
        import statistics as sta
        
        lstm_mu = sta.mean(a)
        bayes_mu = sta.mean(b)
        
        lstm_sd = sta.stdev(a)
        bayes_sd = sta.stdev(b)
        
        
        plt.plot(a)
        plt.xlabel("No. of observations")
        plt.ylabel("Deviations")
        
        plt.plot(b)
        plt.xlabel("No. of observations")
        plt.ylabel("Deviations")


def main():
    # print("Loading Data... Please wait!")
    r_obj = CorrelationCoeff()
    r_obj.importData()
    #r_obj.printAll()
    print("Pearson Correlation", r_obj.perform_correlation())
    print("Spearman Correlation", r_obj.spearman_correlation())
    
    rat, lstm_val, bayes_val = r_obj.getter()
    #plt.plot(lstm_val)
    #plt.plot(bayes_val)
    #plt.plot(rat)
    
    r_obj.ttest()



if __name__ == "__main__":
    with app.app_context():
        main()

# import libraries
import sys
from scipy.stats import pearsonr

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

    # Method for importing dataset
    def importData(self):
        # Import data
        self.__rat_lst = get_actual_sentiments()
        self.__lstm_rat = get_lstm_predictions()
        self.__bayes_rat = get_bayes_predictions()

    def performCorrelation(self):
        lstm_r, _ = pearsonr(self.__rat_lst, self.__lstm_rat)
        bayes_r, _ = pearsonr(self.__rat_lst, self.__bayes_rat)
        return lstm_r, bayes_r

    def printAll(self):
        print(len(self.__lstm_rat), len(self.__bayes_rat))


def main():
    print("Loading Data... Please wait!")
    r_obj = CorrelationCoeff()
    r_obj.importData()
    r_obj.printAll()
    print(r_obj.performCorrelation())


if __name__ == "__main__":
    with app.app_context():
        main()

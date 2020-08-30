# import libraries
import os
import sys

sys.path.append('../')
from bi_lstm_model import BiLSTMModel
from lstm_model import LSTMModel
from naive_bayes_model import NaiveBayesModel
from models import Reviews, db
from ml_utilities import read_data


from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///db.sentiment_analysis'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

with app.app_context():
    db.create_all()

# define class
class DBInitLoad:

    def __init__(self):
        self.__rev_lst, self.__rat_lst = [], []
        self.__lstm_rat, self.__bayes_rat = [], []

    # Method for importing dataset
    def importData(self):
        # Import data

        path = os.path.dirname(__file__)
        dataset_path = path + "/DataSet/"
        test_data_path = dataset_path + "Test/"
        pos_test_data_path = test_data_path + "Positive"
        neg_test_data_path = test_data_path + "Negative"

        self.__rev_lst, self.__rat_lst = read_data(pos_test_data_path, [], [])
        self.__rev_lst, self.__rat_lst = read_data(neg_test_data_path, self.__rev_lst, self.__rat_lst)

    def predictAll(self):
        reviews_to_predict = self.__rev_lst.copy()
        self.__lstm_rat = BiLSTMModel().predict_reviews(self.__rev_lst)
        self.__lstm_rat = [j for sub in self.__lstm_rat for j in sub]
        self.__bayes_rat = NaiveBayesModel().predict_reviews(reviews_to_predict)
        self.__bayes_rat = [j for sub in self.__bayes_rat for j in sub]

    def updateDB(self):
        with app.app_context():
            
            for i in range(len(self.__rev_lst)):
                rev = Reviews.query.get(i+1)        
                rev.review = self.__rev_lst[i]
                rev.actual_sentiment = 0.5
                print(rev.review, rev.actual_sentiment)
                #db.session.query(Reviews).filter_by(id=i+1).update(review=self.__rev_lst[i])
                #db.update(Reviews).values(review=self.__rev_lst[i]).where(Reviews.id==i+1)
                #db.session.commit()
                #update_statement = Reviews.update().values(review=self.__rev_lst[i]).where(Reviews.id==i+1)
                #db.execute(update_statement)
                #r = Reviews(review=self.__rev_lst[i], lstm_prediction=self.__lstm_rat[i], bayes_prediction=self.__bayes_rat[i], actual_sentiment=self.__rat_lst[i])
                #db.session.add(r)
                
            db.session.commit()

    def printAll(self):
        print(self.__rev_lst, self.__rat_lst)


def main():
    print("Loading Data... Please wait!")
    db_obj = DBInitLoad()
    db_obj.importData()
    #db_obj.predictAll()
    db_obj.updateDB()
    #db_obj.printAll()


if __name__ == "__main__":
    main()

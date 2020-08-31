from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import os
from models import db
path = os.path.dirname(__file__)

db = SQLAlchemy()
from models import *

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///db.sentiment_analysis1'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True
db.init_app(app)


# function to get user_id from email_id
def get_user_id(email_id):
    # converting email_id to lowercase as ids are case insensitive
    email_id = email_id.lower()

    # Checking if the user entry is already present add a new entry in users table
    user = User.query.filter_by(email_id=email_id).first()
    if user is None:
        user = User(email_id=email_id)
        db.session.add(user)
        db.session.commit()
    return user.id


# API for frontend: returing dictionery of review ids and respective lstm and bayes predictions
def get_review_details(email_id, reviews):
    """ 
    udhfdsjhfsj
    
    """

    # getting the user_id from email_id
    user_id = get_user_id(email_id)

    # adding reviews to database and creating dictionery to be returned
    model_values = {}
    for review in reviews:
        # TO-DO: code to get lstm prediction
        lstm_prediction = 0.8

        # TO-DO: code to get bayes prediction
        bayes_prediction = 0.7

        # adding new entry in user_reviews table
        r = Reviews(review=review, user_id=user_id, lstm_prediction=lstm_prediction, bayes_prediction=bayes_prediction)
        db.session.add(r)
        db.session.commit()
        # adding key-value pair to dictionery
        model_values[r.id] = [lstm_prediction, bayes_prediction]

    return model_values


# updating the actual user response to the reviews in DB
def update_review_details(actual_values):
    for review_id in actual_values:
        user_review = Reviews.query.get(review_id)
        if user_review is not None:
            user_review.actual_sentiment = actual_values[review_id]
            db.session.commit()

    # trigerring the call to update_user_details function
    update_user_details(1)


# updating the user table with lstm and bayes metric values
def update_user_details(user_id):
    # TO-DO: to be completed once we are sure about the metric
    pass


''' api to update the user ratings based on actual values from user and predicted values from the models'''


def update_user_ratings(user_values):
    for review_id in user_values:
        review = Reviews.query.get(review_id)
        print(review)
        if review is not None:
            lstm_deviation = abs(review.lstm_prediction - review.actual_sentiment)
            bayes_deviation = abs(review.bayes_prediction - review.actual_sentiment)
            user_sentiment = user_values[review_id]
            print(lstm_deviation)
            r = Ratings(lstm_deviation=lstm_deviation, bayes_deviation=bayes_deviation, user_sentiment=user_sentiment,
                        review_id=review_id)
            db.session.add(r)
    db.session.commit()


''' api to get the dictionery of reviews '''


def get_reviews(review_ids):
    review_dict = {}
    for id in review_ids:
        review = Reviews.query.get(id)
        if review is not None:
            review_dict[id] = review.review
    return review_dict


def get_all_reviews():
    reviews = Reviews.query.all()
    review_dict = {}
    for review in reviews:
        review_dict[review.id] = review.review

    return review_dict


def export_csv():
    rev_dct = get_all_reviews()
    df = pd.DataFrame()
    df['id'] = list(rev_dct.keys())
    df['review'] = list(rev_dct.values())
    df.to_csv(path+'/reviews.csv')

def read_csv(lst):
    dataset = pd.read_csv(path+'/reviews.csv')
    # id_lst = dataset.iloc[[i for i in lst], 1].values
    rev_lst = dataset.iloc[[i-1 for i in lst], 2].values
    dct = {}
    for i in range(len(lst)):
        dct[lst[i]] = rev_lst[i]
    return dct


def store_user_rating_to_csv(user_ratings):
    df = pd.DataFrame()
    df['review_id'] = list(user_ratings.keys())
    df['user_ratings'] = list(user_ratings.values())
    df.to_csv(path+'/user_ratings', mode='a', header=False)
    

def get_lstm_predictions():
    lstm_values = list(Reviews.query.with_entities(Reviews.lstm_prediction))
    lstm_values = [val for (val,) in lstm_values]

    return lstm_values


def get_bayes_predictions():
    bayes_values = list(Reviews.query.with_entities(Reviews.bayes_prediction))
    bayes_values = [val for (val,) in bayes_values]

    return bayes_values


def get_actual_sentiments():
    actual_values = list(Reviews.query.with_entities(Reviews.actual_sentiment))
    actual_values = [val for (val,) in actual_values]

    return actual_values


def main():
    # print(get_review_details("u6@gmail.com",["R5","R6"]))
    test_dict = {
        3: 0.65,
        4: 0.70
    }
    # update_user_ratings(test_dict)
    # print(get_reviews([3,4]))
    print(read_csv([222, 1]))
    print(get_reviews([222, 1]))


if __name__ == "__main__":
    with app.app_context():
        main()

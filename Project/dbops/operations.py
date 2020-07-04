from flask import Flask

from models import *

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///db.sentiment_analysis'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
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


# updating the actual user responce to the reviews in DB
def update_review_details(actual_values):
    for review_id in actual_values:
        user_review = Reviews.query.get(review_id)
        user_review.actual_sentiment = actual_values[review_id]
        db.session.commit()

    # trigerring the call to update_user_details function
    update_user_details(1)


# updating the user table with lstm and bayes metric values
def update_user_details(user_id):
    # TO-DO: to be completed once we are sure about the metric
    pass


def main():
    # print(get_review_details("u6@gmail.com",["R5","R6"]))
    test_dict = {
        3: 0.65,
        4: 0.70
    }
    update_review_details(test_dict)


if __name__ == "__main__":
    with app.app_context():
        main()

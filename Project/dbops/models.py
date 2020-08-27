from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Reviews(db.Model):
    __tablename__ = "reviews"
    id = db.Column(db.Integer, primary_key=True)
    review = db.Column(db.String, nullable=False)
    #   user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    lstm_prediction = db.Column(db.Float)
    bayes_prediction = db.Column(db.Float)
    actual_sentiment = db.Column(db.Float)


class Ratings(db.Model):
    __tablename__ = "user_ratings"
    id = db.Column(db.Integer, primary_key=True)
    lstm_deviation = db.Column(db.Float)
    bayes_deviation = db.Column(db.Float)
    user_sentiment = db.Column(db.Float)
    review_id = db.Column(db.Integer, db.ForeignKey("reviews.id"), nullable=False)

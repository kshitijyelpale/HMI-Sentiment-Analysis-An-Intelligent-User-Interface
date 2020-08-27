from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

'''class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email_id = db.Column(db.String(50), unique=True, nullable=False)
    lstm_metric = db.Column(db.Float)
    bayes_metric = db.Column(db.Float)
    reviews = db.relationship("Reviews", backref="user", lazy=True)

    def add_reviews(self, reviews):
        for review in reviews:
            r = Reviews(review=review, user_id=self.id)
            db.session.add(r)
        db.session.commit()'''


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
    bayes_deveation = db.Column(db.Float)
    user_sentiment = db.Column(db.Float)
    review_id = db.Column(db.Integer, db.ForeignKey("reviews.id"), nullable=False)

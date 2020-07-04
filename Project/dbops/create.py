import os

from flask import Flask, render_template, request
from models.Model_LSTM import LSTMModel

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///db.sentiment_analysis'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)


def main():
    db.create_all()


if __name__ == "__main__":
    with app.app_context():
        main()

import sys
from random import seed
from random import randint
sys.path.append(".")
from operations import get_reviews
import streamlit as st
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
app = Flask(__name__)
db.init_app(app)

app = Flask(__name__)
    
seed(1)

email = st.text_input("Enter your email here")

st.text("Please rate following movie reviews")

review_ids = []
for i in range(1,6):
    review_ids.append(randint(1,4501))
    
with app.app_context():
    reviews = get_reviews(review_ids)

user_rating = []
for id in reviews:
    st.text(st.write(reviews[id]))
    user_rating[st.slider("Rate above review", 0.0, 1.0, 0.5, key="r%d" % id)]
    
    
st.write(user_rating)
st.text("Thanks for providing review")

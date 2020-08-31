import sys
from random import seed, randint
sys.path.append(".")
from operations import *
import streamlit as st
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from models import db

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



reviews = read_csv(review_ids)
user_rating = {}
for id in reviews:
    st.text(st.write(reviews[id]))
    st_slidr = st.slider("Rate above review", 0.0, 1.0, 0.5, key="r%d" % id)
    user_rating[id] = st_slidr
    
    
st.text("Thanks for providing review")
submit = st.button("Submit")
if submit:
    st.write(user_rating)
    with app.app_context():
        store_user_rating_to_csv(user_rating)


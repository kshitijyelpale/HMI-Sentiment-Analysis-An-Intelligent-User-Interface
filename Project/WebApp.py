# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 20:55:20 2020

@author: Kshitij
"""
import sys
sys.path.append(".")

import streamlit as st

from models.Model_LSTM_1 import LSTMModel

st.title("LSTM for sentiment analysis")

reviews = []

lstmModel = LSTMModel()

userReview = st.text_area("Enter your review here to detect emotion of it")
st.slider("Rate your review", 0.0, 1.0, 0.5)
#userReview = "great movie"

if userReview != "":
    print(userReview)
    
    reviews.append(userReview)
    max_features = 47000
    max_doc_len = 220
    temp = []
    model = lstmModel.loadModel()
    temp, reviews = lstmModel.preProcessData(temp, reviews)
    temp, reviews = lstmModel.encodeData(max_features, max_doc_len, temp, reviews)
    result = lstmModel.testModel(model, reviews)
    print(result)
    st.write("Emotion is ", result)
    

    

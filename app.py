import streamlit as st
import joblib
import numpy as np

model = joblib.load("model_compressed.pkl")

st.title("Customer Satisfaction Prediction")

response_time = st.number_input("Response Time")
survey_delay = st.number_input("Survey Delay")
sentiment_score = st.number_input("Sentiment Score")

features = np.array([[response_time, survey_delay, sentiment_score]])

if st.button("Predict CSAT"):
    prediction = model.predict(features)
    st.success(f"Predicted CSAT Score: {prediction[0]}")

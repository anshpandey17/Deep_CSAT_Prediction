import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model_compressed.pkl")

st.title("Customer Satisfaction Prediction")

# Inputs
response_time = st.number_input("Response Time")
survey_delay = st.number_input("Survey Delay")
sentiment_score = st.number_input("Sentiment Score")

if st.button("Predict CSAT"):

    # Create base features
    features = np.array([[response_time, survey_delay, sentiment_score]])

    # Get number of features expected by the model
    expected_features = model.n_features_in_

    # Pad remaining features with zeros
    if features.shape[1] < expected_features:
        padding = np.zeros((1, expected_features - features.shape[1]))
        features = np.hstack((features, padding))

    prediction = model.predict(features)

    st.success(f"Predicted CSAT Score: {prediction[0]}")

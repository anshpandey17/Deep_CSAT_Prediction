import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model_compressed.pkl")

st.title("Customer Satisfaction (CSAT) Prediction")

st.write("Enter service metrics to predict CSAT score.")

# Inputs
response_time = st.number_input("Response Time (minutes)", min_value=0.0, value=10.0)
survey_delay = st.number_input("Survey Delay (hours)", min_value=0.0, value=2.0)
sentiment_score = st.number_input("Sentiment Score (-1 to 1)", min_value=-1.0, max_value=1.0, value=0.0)

if st.button("Predict CSAT"):

    # Create full feature array expected by model
    features = np.zeros((1, model.n_features_in_))

    # Fill first 3 features with user inputs
    features[0, 0] = response_time
    features[0, 1] = survey_delay
    features[0, 2] = sentiment_score

    prediction = model.predict(features)

    st.success(f"Predicted CSAT Score: {prediction[0]}")

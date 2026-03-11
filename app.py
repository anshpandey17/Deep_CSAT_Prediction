import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model_compressed.pkl")

st.title("Customer Satisfaction (CSAT) Prediction")

st.write("""
This application predicts **Customer Satisfaction Score (CSAT)** based on key service metrics.
""")

st.subheader("Why These Features Matter")

st.markdown("""
- **Response Time (minutes)** – how quickly support replies  
- **Survey Delay (hours)** – time before customer feedback  
- **Sentiment Score (-1 to 1)** – emotional tone of feedback
""")

st.subheader("Enter Service Metrics")

response_time = st.number_input(
    "Response Time (minutes)",
    min_value=0.0,
    max_value=300.0,
    value=10.0,
    key="rt"
)

survey_delay = st.number_input(
    "Survey Delay (hours)",
    min_value=0.0,
    max_value=72.0,
    value=2.0,
    key="sd"
)

sentiment_score = st.number_input(
    "Sentiment Score (-1 to 1)",
    min_value=-1.0,
    max_value=1.0,
    value=0.0,
    key="ss"
)

if st.button("Predict CSAT"):

    features = np.array([[response_time, survey_delay, sentiment_score]])

    expected_features = model.n_features_in_

    if features.shape[1] < expected_features:
        padding = np.zeros((1, expected_features - features.shape[1]))
        features = np.hstack((features, padding))

    prediction = model.predict(features)

    st.success(f"Predicted CSAT Score: {prediction[0]}")

st.markdown("---")

st.markdown("""
### Model Information
- Model: Random Forest
- Features: Service metrics + sentiment
- Goal: Predict customer satisfaction
""")

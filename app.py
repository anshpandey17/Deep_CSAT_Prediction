import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model_compressed.pkl")

st.title("Customer Satisfaction (CSAT) Prediction")

st.write("""
This application predicts **Customer Satisfaction Score (CSAT)** based on key service metrics.
These features were identified as important during model training.
""")

st.subheader("Why These Features Matter")

st.markdown("""
- **Response Time (minutes)**  
  Measures how quickly customer support responds to a customer issue. Faster responses generally improve satisfaction.

- **Survey Delay (hours)**  
  Indicates the delay between issue resolution and when the customer provides feedback.

- **Sentiment Score (-1 to 1)**  
  Represents the emotional tone of the customer's feedback.
""")

st.subheader("Enter Service Metrics")

response_time = st.number_input(
    "Response Time (minutes)",
    min_value=0.0,
    max_value=300.0,
    value=10.0,
    help="Average time taken by support to respond to the customer (in minutes). Example: 5–30 minutes."
)

survey_delay = st.number_input(
    "Survey Delay (hours)",
    min_value=0.0,
    max_value=72.0,
    value=2.0,
    help="Time between issue resolution and when the customer submitted the feedback survey."
)

sentiment_score = st.number_input(
    "Sentiment Score (-1 to 1)",
    min_value=-1.0,
    max_value=1.0,
    value=0.0,
    help="Sentiment polarity of the customer remark. -1 = negative, 0 = neutral, +1 = positive."
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
### About the Model
- **Model Used:** Random Forest Classifier  
- **Training Data:** Customer support interactions  
- **Features:** Service metrics + sentiment analysis  
- **Goal:** Predict customer satisfaction to help improve support performance
""")
survey_delay = st.number_input(
    "Survey Delay (hours)",
    min_value=0.0,
    max_value=72.0,
    value=2.0,
    help="Time between issue resolution and when the customer submitted the feedback survey."
)

sentiment_score = st.number_input(
    "Sentiment Score (-1 to 1)",
    min_value=-1.0,
    max_value=1.0,
    value=0.0,
    help="Sentiment polarity of the customer remark. -1 = negative, 0 = neutral, +1 = positive."
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
### About the Model
- **Model Used:** Random Forest Classifier  
- **Training Data:** Customer support interactions  
- **Features:** Service metrics + sentiment analysis  
- **Goal:** Predict customer satisfaction to help improve support performance
""")

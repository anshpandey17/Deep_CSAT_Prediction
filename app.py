import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model_compressed.pkl")

# App title
st.title("Customer Satisfaction (CSAT) Prediction")

st.write("""
This application predicts **Customer Satisfaction Score (CSAT)** based on key service metrics.
These features were identified as important during model training.
""")

# Feature explanation
st.subheader("Why These Features Matter")

st.markdown("""
- **Response Time (minutes)**  
  Measures how quickly customer support responds to a customer issue.

- **Survey Delay (hours)**  
  Time between issue resolution and when the customer submits feedback.

- **Sentiment Score (-1 to 1)**  
  Emotional tone of the customer's feedback.
""")

# User input section
st.subheader("Enter Service Metrics")

response_time = st.number_input(
    "Response Time (minutes)",
    min_value=0.0,
    max_value=300.0,
    value=10.0,
    help="Average time taken by support to respond to the customer.",
    key="response_time_input"
)

survey_delay = st.number_input(
    "Survey Delay (hours)",
    min_value=0.0,
    max_value=72.0,
    value=2.0,
    help="Time between issue resolution and when the customer submits feedback.",
    key="survey_delay_input"
)

sentiment_score = st.number_input(
    "Sentiment Score (-1 to 1)",
    min_value=-1.0,
    max_value=1.0,
    value=0.0,
    help="Sentiment polarity of the customer remark.",
    key="sentiment_input"
)

# Prediction button
if st.button("Predict CSAT"):

    # Create feature array
    features = np.array([[response_time, survey_delay, sentiment_score]])

    # Model expects many features (from TF-IDF + tabular features)
    expected_features = model.n_features_in_

    # Pad missing features with zeros
    if features.shape[1] < expected_features:
        padding = np.zeros((1, expected_features - features.shape[1]))
        features = np.hstack((features, padding))

    # Predict
    prediction = model.predict(features)

    st.success(f"Predicted CSAT Score: {prediction[0]}")

# Footer
st.markdown("---")

st.markdown("""
### About the Model
- **Model Used:** Random Forest Classifier  
- **Training Data:** Customer support interaction dataset  
- **Features:** Service metrics + sentiment analysis  
- **Goal:** Predict customer satisfaction to help improve customer support performance
""")

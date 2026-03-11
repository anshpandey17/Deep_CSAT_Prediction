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
- **Response Time**  
  Measures how quickly customer support responds to a customer issue. Faster responses generally improve satisfaction.

- **Survey Delay**  
  Indicates the delay between issue resolution and customer feedback. Longer delays may reduce satisfaction.

- **Sentiment Score**  
  Represents the emotional tone of the customer's feedback. Positive sentiment usually correlates with higher CSAT scores.
""")

st.subheader("Enter Service Metrics")

response_time = st.number_input(
    "Response Time (hours)",
    min_value=0.0,
    help="Average time taken by support agents to respond to the customer."
)

survey_delay = st.number_input(
    "Survey Delay (hours)",
    min_value=0.0,
    help="Time between issue resolution and when the customer submits feedback."
)

sentiment_score = st.number_input(
    "Sentiment Score",
    min_value=-1.0,
    max_value=1.0,
    help="Sentiment polarity of the customer remark (-1 = negative, +1 = positive)."
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
- Model Used: **Random Forest Classifier**
- Features: **Text sentiment + service metrics**
- Purpose: **Predict customer satisfaction to help improve support quality**
""")

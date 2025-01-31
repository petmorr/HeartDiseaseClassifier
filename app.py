import streamlit as st
import requests
import random

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/predict"

# Define feature ranges for randomization
FEATURE_RANGES = {
    "age": (30, 80),
    "sex": (0, 1),
    "chest_pain_type": (1, 4),
    "resting_bp": (90, 180),
    "cholesterol": (120, 400),
    "fasting_blood_sugar": (0, 1),
    "resting_ecg": (0, 2),
    "max_heart_rate": (60, 220),
    "exercise_angina": (0, 1),
    "oldpeak": (0.0, 6.2),
    "st_slope": (0, 2),
}

# Initialize session state for inputs
if "inputs" not in st.session_state:
    st.session_state.inputs = {k: v[0] for k, v in FEATURE_RANGES.items()}

# UI Header
st.title("💓 Heart Disease Prediction")
st.write("Enter patient details or click 'Randomize Input' to generate values.")

# **Randomise Button**
if st.button("🎲 Randomise Input"):
    st.session_state.inputs = {
        feature: random.randint(low, high) if isinstance(low, int) else round(random.uniform(low, high), 2)
        for feature, (low, high) in FEATURE_RANGES.items()
    }

# Create input fields
user_input = {}
for feature, (low, high) in FEATURE_RANGES.items():
    user_input[feature] = st.number_input(
        feature.replace("_", " ").title(),
        min_value=low,
        max_value=high,
        value=st.session_state.inputs[feature],
    )

# **Model Selection**
model_choice = st.selectbox("Select Model:", ["random_forest", "logistic_regression", "svm"])

# **Predict Button**
if st.button("🔍 Predict"):
    response = requests.post(f"{API_URL}/{model_choice}", json=user_input)

    if response.status_code == 200:
        prediction = response.json().get("prediction", "Error")
        st.success(f"🩺 Prediction: **{'Heart Disease Detected' if prediction == 1 else 'No Heart Disease'}**")
    else:
        st.error(f"⚠️ Error: {response.json().get('error', 'Unknown error')}")
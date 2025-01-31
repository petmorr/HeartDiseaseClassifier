import streamlit as st
import requests

# Streamlit UI
st.title("Heart Disease Prediction App")

# Model selection
model_name = st.selectbox("Select Model", ["random_forest", "logistic_regression", "svm"])

# User inputs
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.radio("Sex", [0, 1])
chest_pain_type = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
resting_bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=200)
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=500)
fasting_blood_sugar = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG", [0, 1, 2])
max_heart_rate = st.number_input("Max Heart Rate", min_value=50, max_value=250)
exercise_angina = st.radio("Exercise-Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0)
st_slope = st.selectbox("ST Slope", [0, 1, 2])
target = st.number_input("Target", [0, 1])

# Prediction button
if st.button("Predict"):
    data = {
        "age": age, "sex": sex, "chest_pain_type": chest_pain_type, "resting_bp": resting_bp,
        "cholesterol": cholesterol, "fasting_blood_sugar": fasting_blood_sugar, "resting_ecg": resting_ecg,
        "max_heart_rate": max_heart_rate, "exercise_angina": exercise_angina, "oldpeak": oldpeak,
        "st_slope": st_slope, "target": target
    }

    response = requests.post(f"http://localhost:8000/predict/{model_name}", json=data)

    if response.status_code == 200:
        prediction = response.json()["prediction"]
        result = "Positive (Heart Disease Detected)" if prediction == 1 else "Negative (No Heart Disease)"
        st.success(f"Prediction: {result}")
    else:
        st.error("Error in prediction. Check API connection.")
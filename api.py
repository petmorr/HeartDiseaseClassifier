from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import config
import logging

# Initialize FastAPI
app = FastAPI()

# Load trained models
models = {
    "random_forest": joblib.load(f"{config.MODEL_SAVE_PATH}/random_forest.pkl"),
    "logistic_regression": joblib.load(f"{config.MODEL_SAVE_PATH}/logistic_regression.pkl"),
    "svm": joblib.load(f"{config.MODEL_SAVE_PATH}/svm.pkl")
}

# Define input schema
class InputData(BaseModel):
    age: float
    sex: int
    chest_pain_type: int
    resting_bp: float
    cholesterol: float
    fasting_blood_sugar: int
    resting_ecg: int
    max_heart_rate: float
    exercise_angina: int
    oldpeak: float
    st_slope: int

@app.post("/predict/{model_name}")
def predict(model_name: str, data: InputData):
    """Predict heart disease using a trained model."""

    if model_name not in models:
        logging.error(f"❌ Model {model_name} not found.")
        return {"error": f"Model {model_name} not found"}

    model = models[model_name]

    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # One-Hot Encode categorical variables
    categorical_features = ["chest_pain_type", "resting_ecg", "st_slope"]
    input_df = pd.get_dummies(input_df, columns=categorical_features)

    # Get feature names from the trained model
    trained_feature_names = model.feature_names_in_

    # Ensure all expected training columns exist
    for col in trained_feature_names:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing columns with default value

    # Remove any extra columns not used during training
    input_df = input_df[trained_feature_names]

    # Ensure DataFrame isn't empty
    if input_df.empty:
        logging.error("❌ Input data is empty after processing.")
        return {"error": "Invalid input data. Please check request format."}

    try:
        # Predict using the model
        prediction = model.predict(input_df)[0]
        logging.info(f"✅ Prediction successful for model {model_name}: {prediction}")
        return {"model": model_name, "prediction": int(prediction)}
    except Exception as e:
        logging.error(f"❌ Prediction failed for model {model_name}: {e}")
        return {"error": "Prediction failed. Check model input format."}
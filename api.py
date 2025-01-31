from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import config
import os
import logging
from sklearn.base import BaseEstimator

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] - %(message)s")

# Initialize FastAPI app
app = FastAPI()

# Define model file paths
model_files = {
    "random_forest": f"{config.MODEL_SAVE_PATH}/random_forest.pkl",
    "logistic_regression": f"{config.MODEL_SAVE_PATH}/logistic_regression.pkl",
    "svm": f"{config.MODEL_SAVE_PATH}/svm.pkl"
}

# Load models safely
models = {}
for model_name, file_path in model_files.items():
    if os.path.exists(file_path):
        try:
            model = joblib.load(file_path)
            if isinstance(model, BaseEstimator):  # Ensures it's a scikit-learn model
                models[model_name] = model
                logging.info(f"✅ Loaded {model_name} model successfully from {file_path}")
            else:
                logging.error(f"❌ Invalid model type for {model_name}. Expected scikit-learn model but got {type(model)}.")
        except Exception as e:
            logging.error(f"❌ Failed to load {model_name}: {e}")
    else:
        logging.warning(f"⚠️ Model file not found: {file_path}")

# Load saved feature names from training
feature_names_path = f"{config.MODEL_SAVE_PATH}/feature_names.pkl"
if os.path.exists(feature_names_path):
    feature_names = joblib.load(feature_names_path)
    logging.info("✅ Loaded feature names successfully.")
else:
    logging.error("❌ Feature names file not found. Ensure the model training pipeline saves feature names.")
    feature_names = []

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
        logging.error(f"❌ Model {model_name} not found in available models.")
        return {"error": f"Model {model_name} not found"}

    model = models[model_name]

    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # One-Hot Encode categorical variables
    categorical_features = ["chest_pain_type", "resting_ecg", "st_slope"]
    input_df = pd.get_dummies(input_df, columns=categorical_features)

    # Ensure column order matches training feature names
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing columns with 0 values

    input_df = input_df[feature_names]  # Reorder columns

    try:
        # Make prediction
        prediction = model.predict(input_df)[0]
        logging.info(f"✅ Prediction successful for model {model_name}: {prediction}")
        return {"model": model_name, "prediction": int(prediction)}
    except Exception as e:
        logging.error(f"❌ Prediction failed for model {model_name}: {e}")
        return {"error": "Prediction failed. Check model compatibility and input format."}

@app.get("/")
def root():
    return {"message": "Heart Disease Classifier API is running"}
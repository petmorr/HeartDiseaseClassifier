import pandas as pd
import numpy as np
from logger import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import joblib
import config

def load_and_preprocess_data():
    logger.info("📂 Loading dataset...")
    df = pd.read_csv(config.DATA_PATH)

    # Rename columns
    df.columns = [
        "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
        "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
        "exercise_angina", "oldpeak", "st_slope", "target"
    ]

    # Replace zero values where applicable
    for col in ["resting_bp", "cholesterol", "max_heart_rate"]:
        df[col] = df[col].replace(0, np.nan)

    # Impute missing values with median
    df.fillna(df.median(), inplace=True)

    # 🔹 Prevent log errors by clipping negative `oldpeak` values
    df["oldpeak"] = np.clip(df["oldpeak"], 0, None)  # Set negative values to 0
    df["oldpeak"] = np.log1p(df["oldpeak"])  # Now it's safe to apply log1p

    # Feature Scaling: MinMax for better handling of distributions
    numeric_columns = ["age", "resting_bp", "cholesterol", "max_heart_rate", "oldpeak"]
    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # One-hot encoding categorical features
    categorical_features = ["chest_pain_type", "resting_ecg", "st_slope"]
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # Save feature names
    joblib.dump(df.columns.tolist(), f"{config.MODEL_SAVE_PATH}/feature_names.pkl")

    # Drop highly correlated features (if correlation > 0.9)
    correlation_matrix = df.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]
    df.drop(high_corr_features, axis=1, inplace=True)

    # Splitting dataset into features and labels
    y = df.pop("target")
    X = df

    # 🔹 Dynamically adjust SMOTE sampling strategy
    class_counts = y.value_counts()
    minority_class_size = class_counts.min()
    majority_class_size = class_counts.max()

    # Ensure SMOTE does not try to remove minority samples
    safe_sampling_strategy = min(0.9, minority_class_size / majority_class_size + 0.1)

    logger.info(f"🟢 Applying SMOTE with safe_sampling_strategy={safe_sampling_strategy:.2f}")

    smote = SMOTE(sampling_strategy=safe_sampling_strategy, random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Splitting dataset into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, stratify=y_resampled)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)

    logger.info("✅ Data preprocessing completed successfully.")
    return X_train, X_val, X_test, y_train, y_val, y_test
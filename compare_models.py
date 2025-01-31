import os
import joblib
import pandas as pd
import config
from logger import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.base import BaseEstimator


def load_model(model_path):
    """Safely load a trained model from a file."""
    try:
        model = joblib.load(model_path)
        if isinstance(model, BaseEstimator):  # Ensure it's a valid scikit-learn model
            return model
        else:
            logger.error(f"❌ {model_path} is not a valid scikit-learn model. Got {type(model)} instead.")
            return None
    except Exception as e:
        logger.error(f"❌ Failed to load model from {model_path}: {e}")
        return None


def evaluate_model(model, X_test, y_test):
    """Evaluate a model and return performance metrics."""
    if model is None:
        return {"Accuracy": "N/A", "Precision": "N/A", "Recall": "N/A", "F1 Score": "N/A", "ROC AUC": "N/A"}

    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
        }
        return metrics
    except Exception as e:
        logger.error(f"❌ Error evaluating model: {e}")
        return {"Accuracy": "Error", "Precision": "Error", "Recall": "Error", "F1 Score": "Error", "ROC AUC": "Error"}


def compare_models(X_test, y_test):
    """Load models and compare their performance."""
    logger.info("📊 Starting model comparison...")

    model_paths = [
        os.path.join(config.MODEL_SAVE_PATH, filename)
        for filename in os.listdir(config.MODEL_SAVE_PATH) if filename.endswith(".pkl")
    ]

    models = {os.path.basename(path).replace(".pkl", "").title(): load_model(path) for path in model_paths}

    # Filter out None values (invalid models)
    models = {name: model for name, model in models.items() if model is not None}

    if not models:
        logger.error("❌ No valid models found for comparison.")
        return None

    results = {name: evaluate_model(model, X_test, y_test) for name, model in models.items()}
    df_results = pd.DataFrame.from_dict(results, orient="index")

    logger.info("\n📊 Model Comparison:\n" + df_results.to_string())

    return df_results


if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data

    _, _, X_test, _, _, y_test = load_and_preprocess_data()
    results_df = compare_models(X_test, y_test)
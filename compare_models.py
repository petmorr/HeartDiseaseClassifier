import os
import joblib
import pandas as pd
import config
from logger import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, X_test, y_test):
    """Evaluate a model and return performance metrics."""
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

def compare_models(X_test, y_test):
    """Load models and compare their performance."""
    logger.info("📊 Starting model comparison...")

    models = {
        name: joblib.load(os.path.join(config.MODEL_SAVE_PATH, name))
        for name in os.listdir(config.MODEL_SAVE_PATH) if name.endswith(".pkl")
    }

    results = {name.replace(".pkl", "").title(): evaluate_model(model, X_test, y_test) for name, model in models.items()}
    df_results = pd.DataFrame.from_dict(results, orient="index")

    logger.info("\n📊 Model Comparison:\n" + df_results.to_string())

    return df_results

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data
    _, _, X_test, _, _, y_test = load_and_preprocess_data()
    results_df = compare_models(X_test, y_test)
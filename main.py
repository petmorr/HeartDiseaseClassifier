import config
from data_preprocessing import load_and_preprocess_data
from models.train_random_forest import train_random_forest
from models.train_logistic_regression import train_logistic_regression
from models.train_svm import train_svm
from compare_models import compare_models
from logger import logger
import importlib

# Dynamically load model training modules
MODEL_TRAINING_SCRIPTS = {
    "random_forest": "models.train_random_forest",
    "logistic_regression": "models.train_logistic_regression",
    "svm": "models.train_svm",
    # "xgboost": "models.train_xgboost"  # Uncomment once fixed
}

def main():
    logger.info("🚀 Starting full model training and evaluation pipeline...")

    # Step 1: Data Processing
    try:
        logger.info("📊 Running data processing...")
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()
        logger.info("✅ Data processing completed successfully.")
    except Exception as e:
        logger.error(f"❌ Data processing failed: {e}")
        return

    # Step 2: Train Models Dynamically
    models = {
        "random_forest": train_random_forest,
        "logistic_regression": train_logistic_regression,
        "svm": train_svm
    }

    for model_name, model_func in models.items():
        try:
            logger.info(f"🤖 Training {model_name} model...")
            model_func(X_train, X_val, X_test, y_train, y_val, y_test)  # Call correct function
        except Exception as e:
            logger.error(f"❌ {model_name} training failed: {e}")

    logger.info("✅ All models trained successfully.")

    # Step 3: Compare Model Performance
    try:
        logger.info("📊 Running model comparison...")
        results_df = compare_models(X_test, y_test)
        best_model_name = results_df["Accuracy"].idxmax()
        best_model_accuracy = results_df.loc[best_model_name, "Accuracy"]
        logger.info(f"🏆 Best Model: {best_model_name} with Accuracy: {best_model_accuracy:.4f}")
        print(f"\n🏆 Best Model: {best_model_name} with Accuracy: {best_model_accuracy:.4f}")
    except Exception as e:
        logger.error(f"❌ Model comparison failed: {e}")

    logger.info("✅ Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
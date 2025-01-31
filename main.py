import config
from data_preprocessing import load_and_preprocess_data
from models.train_random_forest import train_random_forest
from models.train_logistic_regression import train_logistic_regression
from models.train_svm import train_svm
from compare_models import compare_models
from logger import logger
import importlib

# Define available models
MODEL_TRAINING_FUNCTIONS = {
    "logistic_regression": train_logistic_regression,
    "random_forest": train_random_forest,
    "svm": train_svm
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

    # Step 2: Train Models
    for model_name, train_func in MODEL_TRAINING_FUNCTIONS.items():
        try:
            logger.info(f"🤖 Training {model_name} model...")
            train_func(X_train, X_val, X_test, y_train, y_val, y_test)
            logger.info(f"✅ {model_name} model trained successfully.")
        except Exception as e:
            logger.error(f"❌ {model_name} training failed: {e}")

    logger.info("✅ All models trained successfully.")

    # Step 3: Compare Model Performance
    try:
        logger.info("📊 Running model comparison...")
        results_df = compare_models(X_test, y_test)

        if results_df is not None and not results_df.empty:
            best_model_name = results_df["Accuracy"].idxmax()
            best_model_accuracy = results_df.loc[best_model_name, "Accuracy"]
            logger.info(f"🏆 Best Model: {best_model_name} with Accuracy: {best_model_accuracy:.4f}")
            print(f"\n🏆 Best Model: {best_model_name} with Accuracy: {best_model_accuracy:.4f}")
        else:
            logger.warning("⚠️ No valid models available for comparison.")
    except Exception as e:
        logger.error(f"❌ Model comparison failed: {e}")

    logger.info("✅ Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
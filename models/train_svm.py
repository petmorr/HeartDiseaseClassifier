from logger import logger
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import config

def train_svm(X_train, X_val, X_test, y_train, y_val, y_test):
    logger.info("🔍 Training SVM model...")

    try:
        # Perform hyperparameter tuning using RandomizedSearchCV for efficiency
        random_search = RandomizedSearchCV(
            estimator=SVC(probability=True, random_state=42),
            param_distributions=config.HYPERPARAMS["svm"],
            cv=10, scoring="accuracy", n_jobs=-1, verbose=3, n_iter=30, random_state=42
        )

        logger.info("🔬 Performing hyperparameter tuning...")
        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        logger.info(f"✅ Best Parameters: {random_search.best_params_}")

        # Validate the model
        y_val_pred = best_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        logger.info(f"📊 Validation Accuracy: {val_accuracy:.4f}")

        # Evaluate on Test Set
        y_test_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_report = classification_report(y_test, y_test_pred)

        logger.info(f"📈 Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"📜 Classification Report:\n{test_report}")

        # Save the best model
        joblib.dump(best_model, f"{config.MODEL_SAVE_PATH}/svm.pkl")
        logger.info("💾 SVM model saved successfully.")

    except Exception as e:
        logger.error(f"❌ SVM training failed: {e}")
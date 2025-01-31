from logger import logger
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import config

def get_valid_param_grid():
    """Generate a valid parameter grid for Logistic Regression."""
    param_grid = []
    for solver, penalties in config.HYPERPARAMS["logistic_regression"]["solver_penalty"].items():
        for penalty in penalties:
            params = {
                "solver": [solver],
                "penalty": [penalty],
                "C": config.HYPERPARAMS["logistic_regression"]["C"]
            }
            if penalty == "elasticnet" and solver == "saga":
                params["l1_ratio"] = config.HYPERPARAMS["logistic_regression"]["l1_ratio"]
            param_grid.append(params)
    return param_grid

def train_logistic_regression(X_train, X_val, X_test, y_train, y_val, y_test):
    logger.info("🔍 Training Logistic Regression model...")

    try:
        # Define model
        lr = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)

        # Generate a valid hyperparameter grid
        param_grid = get_valid_param_grid()

        # Hyperparameter tuning using RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=lr,
            param_distributions=param_grid,
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

        # Save the model
        joblib.dump(best_model, f"{config.MODEL_SAVE_PATH}/logistic_regression.pkl")
        logger.info("💾 Logistic Regression model saved successfully.")

    except Exception as e:
        logger.error(f"❌ Logistic Regression training failed: {e}")
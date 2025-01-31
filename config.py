import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "heart_statlog_cleveland_hungary_final.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models")
LOG_PATH = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
for path in [MODEL_SAVE_PATH, LOG_PATH]:
    os.makedirs(path, exist_ok=True)

# Hyperparameters for different models
HYPERPARAMS = {
    "random_forest": {
        "n_estimators": [100, 300, 500, 1000],
        "max_depth": [None, 10, 20, 50, 100],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "bootstrap": [True, False],
        "class_weight": ["balanced", "balanced_subsample", None]
    },
    "logistic_regression": {
        "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        "solver_penalty": {
            "lbfgs": ["l2"],
            "liblinear": ["l1", "l2"],
            "newton-cg": ["l2"],
            "sag": ["l2"],
            "saga": ["l1", "l2", "elasticnet"]
        },
        "l1_ratio": [0.1, 0.5, 0.9]
    },
    "svm": {
        "C": [0.01, 0.1, 1, 10, 100, 1000],
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
        "degree": [2, 3, 4],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
        "shrinking": [True, False]
    }
}
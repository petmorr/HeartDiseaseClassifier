# 💖 Heart Disease Classifier

## 📌 Overview
The **Heart Disease Classifier** is a machine learning-based web application designed to predict heart disease risk based on user input data. It integrates multiple models, advanced preprocessing, and a user-friendly UI for seamless interaction. The backend is powered by **FastAPI**, while the UI is built with **Streamlit**.

---

## 🚀 Features
✅ **Multiple ML Models:** Supports **Random Forest, Logistic Regression, and SVM**  
✅ **FastAPI Backend:** REST API for model inference  
✅ **Streamlit UI:** Interactive web-based interface for user predictions  
✅ **SMOTE Handling:** Balances dataset to improve model performance  
✅ **Hyperparameter Optimization:** Fine-tuned models for better accuracy  
✅ **Logging:** Centralized logging for debugging and tracking  
✅ **Docker Ready:** Easily deployable with containerization  

---

## 📂 Project Structure

```
HeartDiseaseClassifier/ 
│── models/ # Trained models (.pkl files) 
    │── train_random_forest.py # Random Forest training script 
    │── train_logistic_regression.py # Logistic Regression training script 
    │── train_svm.py # SVM training script
│── logs/ # Log files for debugging 
│── data/ # Dataset storage 
│── api.py # FastAPI backend for predictions 
│── app.py # Streamlit UI 
│── config.py # Configuration file 
│── main.py # Main script to train and evaluate models 
│── logger.py # Centralized logging setup 
│── data_preprocessing.py # Data cleaning & feature engineering 
│── compare_models.py # Model evaluation and comparison  
│── requirements.txt # Project dependencies 
│── README.md # Documentation (this file)
```

---

## 🛠️ Installation Guide

### 📌 Prerequisites
Ensure you have the following installed:
- **Python 3.8+**
- **pip**
- **virtualenv (optional)**
- **Docker (optional for containerization)**

### 🚀 Setup Steps
1. **Clone the repository**  
   ```sh
   git clone https://github.com/your-repo/HeartDiseaseClassifier.git
   cd HeartDiseaseClassifier
   ```
2. **Create a virtual environment (Optional but recommended)**
    ```shell
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\\Scripts\\activate     # On Windows
    ```
3. **Install dependencies**
    ```shell
    pip install -r requirements.txt
    ```
4. **Run model training (if models are not pre-trained)**
    ```sh
    python main.py
    ```
5. Start the FastAPI server
    ```sh 
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
    ```
API will be available at:
📌 http://127.0.0.1:8000/docs

6. Run the Streamlit UI
    ```sh
    streamlit run app.py
    ```
UI will be available at:
📌 http://localhost:8501

---

# 🔬 Model Training

* The dataset undergoes preprocessing (handling missing values, encoding, normalization).
* SMOTE is applied to balance class distribution.
* Hyperparameter tuning is performed using GridSearchCV.
* The models are trained and saved as .pkl files for deployment.


---

# 📊 Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|------------|--------|----------|
| Random Forest      | 89.5%    | 0.94       | 0.84   | 0.89     |
| Logistic Regression | 81.5%    | 0.82       | 0.80   | 0.81     |
| SVM                | 88.9%    | 0.93       | 0.84   | 0.88     |

---

# 🐳 Docker Deployment

To deploy the application using Docker, follow these steps:

1. Build the Docker Image
    ```sh
   docker build -t heart-disease-classifier . 
    ```
2. Run the Docker Container
    ```sh
    docker run -p 8000:8000 -p 8501:8501 heart-disease-classifier
    ```
3. Access the API & UI
   * API: http://localhost:8000/docs
   * UI: http://localhost:8501

--- 

# 📌 Future Improvements
🔹 Improve hyperparameter tuning for better accuracy

🔹 Implement additional ML models like XGBoost or Neural Networks

🔹 Integrate a database for real-time patient data storage

🔹 Deploy the API and UI to cloud platforms (AWS, Azure, GCP)

---
# 📄 License
This project is **open-source** and available under the **MIT License**.
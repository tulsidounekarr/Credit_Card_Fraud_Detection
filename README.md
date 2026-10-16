# 💳 Credit Card Fraud Detection System

An AI-powered credit card fraud detection project built with **Python, Machine Learning, XGBoost, and Streamlit**. The system analyzes transaction data, detects potentially fraudulent activity, compares classification models, and presents results through an interactive dashboard.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Model-XGBoost-red" alt="XGBoost">
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Status-Educational%20Project-success" alt="Project Status">
</p>

---

## ✨ Features

- 💳 **Transaction Classification** — Predict whether a transaction may be fraudulent or genuine.
- 🤖 **Machine Learning Models** — Train and compare Random Forest and XGBoost classifiers.
- ⚖️ **Class Imbalance Handling** — Apply SMOTE to balance the training data.
- 🔍 **Anomaly Detection** — Use Isolation Forest and Local Outlier Factor (LOF).
- 📊 **Interactive Dashboard** — Enter transaction features and view prediction results.
- 📈 **ROC Curve Analysis** — Compare classification performance using ROC-AUC.
- 🎯 **Confusion Matrix** — Visualize correct predictions and classification errors.
- 💾 **Model Persistence** — Save trained models and preprocessing objects using Joblib.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data processing and numerical analysis |
| Scikit-learn | Machine learning, preprocessing, and evaluation |
| XGBoost | Gradient-boosted classification |
| imbalanced-learn | SMOTE oversampling |
| Isolation Forest | Unsupervised anomaly detection |
| Local Outlier Factor | Local anomaly detection |
| Streamlit | Interactive web dashboard |
| Matplotlib | ROC curves and confusion matrix visualization |
| Joblib | Saving and loading trained models |

## 📸 Project Screenshots

### 1. 💳 Fraud Prediction Result

View the estimated fraud probability and the model's classification for a transaction.

![Fraud Prediction Result](screenshots/01_fraud_prediction_result.png)

### 2. 📈 Model Evaluation Dashboard

Compare Random Forest and XGBoost using ROC curves and ROC-AUC scores.

![Model Evaluation Dashboard](screenshots/02_model_evaluation_dashboard.png)

### 3. 🖥️ Transaction Input Dashboard

Enter transaction details through the Streamlit interface.

![Transaction Input Dashboard](screenshots/03_transaction_input_dashboard.png)

### 4. 🔢 Anonymized Transaction Features

Enter the Time, Amount, and anonymized V1–V28 features required by the trained model.

![Transaction Features Form](screenshots/04_transaction_features_form.png)

## 📊 Model Performance

The following results were obtained during one training run using a stratified train/test split and SMOTE applied to the training set.

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| Fraud Precision | 0.87 | 0.36 |
| Fraud Recall | 0.83 | 0.87 |
| Fraud F1-score | 0.85 | 0.51 |
| ROC-AUC | 0.9684 | **0.9800** |

**Interpretation:**
- Random Forest achieved a better precision–recall balance at the current classification threshold.
- XGBoost achieved a higher ROC-AUC and

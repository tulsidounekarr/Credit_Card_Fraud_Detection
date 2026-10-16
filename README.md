💳 Credit Card Fraud Detection System

An AI-powered credit card fraud detection project built with Python, Machine Learning, XGBoost, and Streamlit. The system analyzes transaction data, detects potentially fraudulent activity, compares classification models, and presents results through an interactive dashboard.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Model-XGBoost-red" alt="XGBoost">
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Status-Educational%20Project-success" alt="Project Status">
</p>

✨ Features

💳 Transaction Classification — Predict whether a transaction may be fraudulent or genuine.

🤖 Machine Learning Models — Train and compare Random Forest and XGBoost classifiers.

⚖️ Class Imbalance Handling — Apply SMOTE to balance the training data.

🔍 Anomaly Detection — Use Isolation Forest and Local Outlier Factor (LOF).

📊 Interactive Dashboard — Enter transaction features and view prediction results.

📈 ROC Curve Analysis — Compare classification performance using ROC-AUC.

🎯 Confusion Matrix — Visualize correct predictions and classification errors.

💾 Model Persistence — Save trained models and preprocessing objects using Joblib.

🛠️ Tech Stack

Technology

Purpose

Python

Core programming language

Pandas & NumPy

Data processing and numerical analysis

Scikit-learn

Machine learning, preprocessing, and evaluation

XGBoost

Gradient-boosted classification

imbalanced-learn

SMOTE oversampling

Isolation Forest

Unsupervised anomaly detection

Local Outlier Factor

Local anomaly detection

Streamlit

Interactive web dashboard

Matplotlib

ROC curves and confusion matrix visualization

Joblib

Saving and loading trained models

📸 Project Screenshots

1. 💳 Fraud Prediction Result

View the estimated fraud probability and the model's classification for a transaction.



2. 📈 Model Evaluation Dashboard

Compare Random Forest and XGBoost using ROC curves and ROC-AUC scores.



3. 🖥️ Transaction Input Dashboard

Enter transaction details through the Streamlit interface.



4. 🔢 Anonymized Transaction Features

Enter the Time, Amount, and anonymized V1–V28 features required by the trained model.



📊 Model Performance

The following results were obtained during one training run using a stratified train/test split and SMOTE applied to the training set.

Metric

Random Forest

XGBoost

Fraud Precision

0.87

0.36

Fraud Recall

0.83

0.87

Fraud F1-score

0.85

0.51

ROC-AUC

0.9684

0.9800

Interpretation:

Random Forest achieved a better precision–recall balance at the current classification threshold.

XGBoost achieved a higher ROC-AUC and detected a greater proportion of fraudulent transactions, but produced more false positives at the selected threshold.

The dataset is highly imbalanced, so accuracy alone is not sufficient to evaluate fraud detection.

Results may vary with different data splits, parameters, and decision thresholds.

📁 Project Structure

Credit_Card_Fraud_Detection/
├── dataset/
│   └── creditcard.csv              # Download separately
├── outputs/
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   ├── scaler.pkl                  # Generated during training
│   ├── xgboost_model.pkl            # Generated during training
│   ├── isolation_forest.pkl         # Generated during training
│   └── lof_model.pkl                # Generated during training
├── screenshots/
│   ├── 01_fraud_prediction_result.png
│   ├── 02_model_evaluation_dashboard.png
│   ├── 03_transaction_input_dashboard.png
│   └── 04_transaction_features_form.png
├── advanced_fraud_detection.py
├── app.py
├── credit_card_fraud_detection.py
├── requirements.txt
├── README.md
├── Class_Distribution.png
└── Confusion_Matrix.png

🚀 Installation & Setup

1. Clone the repository

git clone https://github.com/tulsidounekarr/Credit_Card_Fraud_Detection.git
cd Credit_Card_Fraud_Detection

2. Create a virtual environment (recommended)

python -m venv .venv
.\.venv\Scripts\Activate.ps1

3. Install dependencies

python -m pip install -r requirements.txt

If any required packages are missing, install them with:

python -m pip install pandas numpy scikit-learn xgboost imbalanced-learn streamlit matplotlib joblib

4. Download the dataset

Download the Credit Card Fraud Detection dataset from Kaggle:

Download Dataset — Kaggle

Extract creditcard.csv and place it here:

dataset/creditcard.csv

Keep the dataset local; do not commit it to a public repository.

5. Train the models

python advanced_fraud_detection.py

This trains the supervised models, fits the anomaly detectors, and generates evaluation charts and model files in outputs/.

6. Launch the dashboard

python -m streamlit run app.py

Open the local URL shown in the terminal, usually:

http://localhost:8501

📉 Evaluation Metrics

Precision: How many transactions predicted as fraud were actually fraudulent.

Recall: How many actual fraudulent transactions the model detected.

F1-score: The harmonic mean of precision and recall.

ROC-AUC: How well the model distinguishes fraudulent transactions from genuine ones across classification thresholds.

Confusion Matrix: Counts true positives, true negatives, false positives, and false negatives.

🔐 Data & Model Considerations

The Kaggle dataset contains anonymized transaction features, including V1 through V28.

The dashboard is intended for experimentation with the dataset's feature format, not arbitrary real-world card details.

The anomaly detectors are trained and saved separately; the current dashboard uses the XGBoost classifier for predictions.

Trained .pkl files are generated locally and do not need to be committed to GitHub.

For real deployment, add input validation, monitoring, model versioning, security controls, and threshold calibration.

🎯 Future Improvements

Add a dedicated anomaly-detection comparison page.

Add model selection and adjustable classification thresholds.

Display detailed prediction explanations and feature importance.

Add batch transaction analysis through CSV upload.

Deploy the Streamlit application to a hosting platform.

Add automated tests and a reproducible training pipeline.

⚠️ Disclaimer

This project is intended for educational and demonstration purposes only. Predictions are estimates and may be incorrect. This application is not a production banking fraud detection system and should not be used to make real financial decisions.

👨‍💻 Author

Tulsi Dounekar

GitHub: @tulsidounekarr

<p align="center">
  ⭐ If you find this project useful, consider giving the repository a star!
</p>

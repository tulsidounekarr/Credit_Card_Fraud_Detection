# Credit Card Fraud Detection using Machine Learning

## 📌 Project Overview

Credit Card Fraud Detection is a Machine Learning project that identifies fraudulent transactions from credit card transaction data.

The main challenge in fraud detection is handling highly imbalanced data, where genuine transactions are much higher than fraudulent transactions. This project uses **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset and improves fraud detection performance using a **Random Forest Classifier**.

---

## 🎯 Objectives

- Analyze credit card transaction data
- Identify fraudulent transactions using Machine Learning
- Handle class imbalance problem
- Train and evaluate a classification model
- Measure model performance using evaluation metrics

---

## 📂 Dataset Information

Dataset contains credit card transactions with:

- **284,807 total transactions**
- **30 input features**
- **1 target variable (Class)**

Target variable:

| Class | Meaning |
|------|---------|
| 0 | Genuine Transaction |
| 1 | Fraudulent Transaction |

Dataset distribution:

- Genuine transactions: 284,315
- Fraud transactions: 492

The dataset is highly imbalanced, with fraud cases representing only **0.17%** of total transactions.

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Joblib

---

## 🔍 Project Workflow

1. Import required libraries
2. Load credit card transaction dataset
3. Perform exploratory data analysis
4. Check missing values and data distribution
5. Visualize fraud vs genuine transactions
6. Perform feature scaling
7. Split dataset into training and testing sets
8. Handle class imbalance using SMOTE
9. Train Random Forest Classification model
10. Evaluate model performance
11. Save trained model

---

## 🤖 Machine Learning Model

### Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

Model parameters:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

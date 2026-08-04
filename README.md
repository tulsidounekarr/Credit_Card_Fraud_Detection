# 💳 Credit Card Fraud Detection using Machine Learning

## 📌 Project Overview

This project detects fraudulent credit card transactions using Machine Learning techniques. The model is trained on the **Credit Card Fraud Detection Dataset** and learns patterns from transaction details to classify transactions as **legitimate or fraudulent**.

The project demonstrates a complete Machine Learning workflow including data exploration, preprocessing, handling class imbalance, model training, evaluation, visualization, and saving the trained model.

---

## 🚀 Features

- Load and analyze credit card transaction dataset
- Perform Exploratory Data Analysis (EDA)
- Check missing values and data distribution
- Handle highly imbalanced dataset
- Preprocess transaction data
- Train Machine Learning classification models
- Detect fraudulent transactions
- Evaluate model performance
- Generate confusion matrix visualization
- Save trained model using Joblib
- Predict fraud probability for new transactions

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

---

## 📂 Project Structure

```
Credit_Card_Fraud_Detection/
│
├── dataset/
│   └── creditcard.csv
│
├── credit_card_fraud_detection.py
├── fraud_detection_model.pkl
├── confusion_matrix.png
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 📊 Dataset

### Dataset Name:
**Credit Card Fraud Detection Dataset**

The dataset contains transactions made by credit cards. It includes numerical features generated using PCA transformation.

### Dataset Features:

- Time
- V1 - V28 (PCA transformed features)
- Amount
- Class (Target Variable)

### Target Variable:

```
Class
```

- `0` → Normal Transaction
- `1` → Fraudulent Transaction

---

# ⚙️ Machine Learning Workflow

## Step 1: Import Libraries

Libraries used:

- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

---

## Step 2: Load Dataset

The credit card transaction dataset is loaded using Pandas.

Example:

```python
df = pd.read_csv("dataset/creditcard.csv")
```

---

## Step 3: Data Exploration

Performed analysis:

- Display first few records
- Check dataset shape
- Analyze data types
- Check missing values
- Study class distribution

---

## Step 4: Data Preprocessing

Steps performed:

- Check and remove missing values
- Separate features and target variable
- Normalize transaction amount values
- Prepare data for model training

---

## Step 5: Handling Class Imbalance

The dataset contains a very small number of fraudulent transactions compared to normal transactions.

Techniques used:

- Stratified Train-Test Split
- Class Weight Handling

---

## Step 6: Train-Test Split

Dataset divided into:

- **80% Training Data**
- **20% Testing Data**

---

## Step 7: Model Training

Machine Learning model used:

### Random Forest Classifier

Random Forest is used because it performs well on classification problems and handles complex patterns effectively.

---

## Step 8: Model Evaluation

The model is evaluated using:

- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- Confusion Matrix

---

## Step 9: Visualization

Generated visualization:

### Confusion Matrix

Output File:

```
confusion_matrix.png
```

The visualization shows:

- Correct predictions
- Incorrect predictions
- Fraud detection performance

---

## Step 10: Save Model

The trained model is saved using Joblib.

Saved Model:

```
fraud_detection_model.pkl
```

---

# 📈 Model Performance

### Evaluation Metrics:

| Metric | Score |
|--------|-------|
| Accuracy | 99%+ |
| Precision | High |
| Recall | High |
| F1 Score | High |

The model achieves strong performance in identifying fraudulent transactions while minimizing false predictions.

---

# 🎯 Fraud Prediction Example

### Example Input:

| Feature | Value |
|---------|-------|
| Time | 50000 |
| Amount | 250 |
| V1 | -1.35 |
| V2 | 0.45 |
| V3 | 1.20 |

### Prediction Output:

```
Transaction Status: Fraud
```

or

```
Transaction Status: Normal
```

---

# ▶️ How to Run the Project

## Clone Repository

```bash
git clone https://github.com/yourusername/Credit_Card_Fraud_Detection.git
```

---

## Move to Project Folder

```bash
cd Credit_Card_Fraud_Detection
```

---

## Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python credit_card_fraud_detection.py
```

---

# 📁 Output Files

The project generates:

```
credit_card_fraud_detection.py
fraud_detection_model.pkl
confusion_matrix.png
README.md
requirements.txt
```

---

# 💡 Future Improvements

- Handle class imbalance using SMOTE technique
- Try advanced models like:
  - XGBoost
  - LightGBM
  - Neural Networks
- Build an interactive Streamlit web application
- Add real-time fraud detection API
- Deploy the model using cloud platforms

---

# 👩‍💻 Author

**Tulsi Dounekar**

GitHub:
https://github.com/tulsidounekarr

---

## ⭐ If you found this project helpful

Please consider giving this repository a ⭐ on GitHub!

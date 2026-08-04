import pandas as pd

# Load Dataset
df = pd.read_csv("dataset/creditcard.csv", low_memory=False)
# Display first 5 rows
print("\nFirst 5 Rows of Dataset:")
print(df.head())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Data types
print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

#Check Class Distribution
print("\nClass Distribution:")
print(df["Class"].value_counts())

print("\nPercentage Distribution:")
print(df["Class"].value_counts(normalize=True) * 100)

# Step 4: Visualize Class Distribution
import matplotlib.pyplot as plt

class_counts = df["Class"].value_counts()

plt.figure(figsize=(6,5))
plt.bar(class_counts.index.astype(str), class_counts.values)

plt.title("Credit Card Transaction Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Transactions")
plt.xticks(["0", "1"], ["Genuine (0)", "Fraud (1)"])

plt.grid(axis="y")

# Save graph
plt.savefig("Class_Distribution.png")

plt.close()

print("\nClass Distribution Graph Saved Successfully!")

# Step 3: Check Class Distribution 

print("\nClass Distribution:")
print(df["Class"].value_counts())

# Step 4: Visualize Class Distribution

import matplotlib.pyplot as plt

class_counts = df["Class"].value_counts()

plt.figure(figsize=(6,5))

plt.bar(
    ["Genuine (0)", "Fraud (1)"],
    class_counts.values
)

plt.title("Credit Card Fraud Distribution")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

plt.savefig("Class_Distribution.png")

print("\nClass Distribution Graph Saved Successfully!")

# Step 5: Feature Selection

X = df.drop("Class", axis=1)
y = df["Class"]


# Step 6: Train Test Split

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nBefore SMOTE:")
print(y_train.value_counts())


# Handle imbalance

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)


print("\nAfter SMOTE:")
print(y_train.value_counts())


# Step 7: Random Forest Model

from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


print("\nTraining Started...")

model.fit(X_train, y_train)

print("Model Training Completed!")


# Step 8: Prediction

y_pred = model.predict(X_test)


# Step 9: Evaluation

from sklearn.metrics import classification_report, confusion_matrix


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Step 10: Save Model

import joblib

joblib.dump(
    model,
    "credit_card_fraud_model.pkl"
)


print("\nModel Saved Successfully!")

print("\nCredit Card Fraud Detection Completed Successfully!")


#
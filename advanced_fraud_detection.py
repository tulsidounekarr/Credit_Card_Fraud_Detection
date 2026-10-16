
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# 1. Load dataset
DATA_PATH = "dataset/creditcard.csv"
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Class"])
y = df["Class"]

print("Dataset shape:", df.shape)
print("\nClass distribution:\n", y.value_counts())

# 2. Split first to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Fit scaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Balance training data only
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_scaled, y_train
)

# 5. Train supervised classifiers
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_balanced, y_train_balanced)

    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    results[name] = {
        "model": model,
        "predictions": predictions,
        "probabilities": probabilities,
        "auc": roc_auc_score(y_test, probabilities)
    }

    print(f"\n{name} classification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("ROC-AUC:", round(results[name]["auc"], 4))

# 6. Plot ROC curves
plt.figure(figsize=(8, 6))

for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result["probabilities"])
    plt.plot(
        fpr, tpr,
        label=f'{name} (AUC={result["auc"]:.3f})'
    )

plt.plot([0, 1], [0, 1], "k--", label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Credit Card Fraud Detection - ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/roc_curve.png", dpi=150)
plt.close()

# 7. Save XGBoost confusion matrix
best_name = max(results, key=lambda name: results[name]["auc"])
best_result = results[best_name]

ConfusionMatrixDisplay.from_predictions(
    y_test, best_result["predictions"],
    display_labels=["Genuine", "Fraud"],
    values_format="d"
)
plt.title(f"{best_name} - Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()

# 8. Train anomaly detectors on training data
# Use a sample for speed; do not use test labels to fit detectors.
sample_size = min(50000, len(X_train_scaled))
anomaly_sample = X_train_scaled[:sample_size]

isolation_forest = IsolationForest(
    contamination=0.002,
    random_state=42,
    n_jobs=-1
)
isolation_forest.fit(anomaly_sample)

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.002,
    novelty=True,
    n_jobs=-1
)
lof.fit(anomaly_sample)

# 9. Save artifacts for later dashboard use
joblib.dump(scaler, "outputs/scaler.pkl")
joblib.dump(results["XGBoost"]["model"], "outputs/xgboost_model.pkl")
joblib.dump(isolation_forest, "outputs/isolation_forest.pkl")
joblib.dump(lof, "outputs/lof_model.pkl")

print("\nSaved ROC curve and confusion matrix in outputs/.")
print("Saved scaler and trained models in outputs/.")
print("Anomaly detectors trained successfully.")

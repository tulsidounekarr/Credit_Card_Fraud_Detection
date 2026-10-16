
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")
st.write("AI-powered transaction fraud prediction dashboard")

MODEL_PATH = "outputs/xgboost_model.pkl"
SCALER_PATH = "outputs/scaler.pkl"
DATA_PATH = "dataset/creditcard.csv"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Model files not found. Run advanced_fraud_detection.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.info(
    "Enter the transaction's Time, Amount, and V1–V28 values. "
    "These fields must use the same feature format as the training dataset."
)

with st.expander("Try an example from the dataset"):
    if os.path.exists(DATA_PATH):
        sample_df = pd.read_csv(DATA_PATH)
        example_index = st.number_input(
            "Dataset row number",
            min_value=0,
            max_value=len(sample_df) - 1,
            value=0,
            step=1
        )
        if st.button("Load example"):
            st.session_state["example"] = sample_df.iloc[int(example_index)]

feature_names = (
    ["Time"] +
    [f"V{i}" for i in range(1, 29)] +
    ["Amount"]
)

example = st.session_state.get("example")

with st.form("transaction_form"):
    st.subheader("Transaction details")

    time_value = st.number_input(
        "Time (seconds from dataset start)",
        min_value=0.0,
        value=float(example["Time"]) if example is not None else 0.0
    )
    amount_value = st.number_input(
        "Transaction amount",
        min_value=0.0,
        value=float(example["Amount"]) if example is not None else 100.0
    )

    st.subheader("Anonymized transaction features")
    values = {}

    columns = st.columns(4)
    for i in range(1, 29):
        feature = f"V{i}"
        with columns[(i - 1) % 4]:
            values[feature] = st.number_input(
                feature,
                value=float(example[feature]) if example is not None else 0.0,
                format="%.6f"
            )

    submitted = st.form_submit_button("Analyze Transaction")

if submitted:
    row = {"Time": time_value}
    row.update(values)
    row["Amount"] = amount_value

    input_df = pd.DataFrame([row], columns=feature_names)
    scaled_input = scaler.transform(input_df)

    probability = float(model.predict_proba(scaled_input)[0][1])
    prediction = int(model.predict(scaled_input)[0])

    st.subheader("Analysis result")
    st.metric("Estimated fraud probability", f"{probability:.2%}")

    if prediction == 1:
        st.error("⚠️ The model classified this transaction as potentially fraudulent.")
    else:
        st.success("✅ The model classified this transaction as likely genuine.")

    st.caption(
        "This is an educational machine-learning demonstration, "
        "not a guarantee of whether a real transaction is fraudulent."
    )

st.divider()
st.subheader("Model evaluation")

roc_path = "outputs/roc_curve.png"
cm_path = "outputs/confusion_matrix.png"

left, right = st.columns(2)

with left:
    if os.path.exists(roc_path):
        st.image(roc_path, caption="ROC curve comparison", use_container_width=True)

with right:
    if os.path.exists(cm_path):
        st.image(cm_path, caption="XGBoost confusion matrix", use_container_width=True)

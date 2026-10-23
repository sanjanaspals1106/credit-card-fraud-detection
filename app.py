from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.joblib"

st.set_page_config(
    page_title="CardGuard | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ CardGuard")
st.subheader("Credit Card Fraud Detection")
st.caption("Machine learning risk assessment — educational project")

if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.info(
        "Run the model-saving cell in your Jupyter notebook and save "
        "fraud_detection_model.joblib inside the models folder."
    )
    st.stop()

@st.cache_resource
def load_artifact(path):
    return joblib.load(path)

artifact = load_artifact(str(MODEL_PATH))

model = artifact["model"]
threshold = float(artifact.get("threshold", 0.5))
feature_columns = artifact["feature_columns"]
model_name = artifact.get("model_name", type(model).__name__)

st.sidebar.header("Model information")
st.sidebar.write(f"**Model:** {model_name}")
st.sidebar.write(f"**Threshold:** {threshold:.4f}")
st.sidebar.warning("Educational use only. Not for real payment decisions.")

st.write(
    "Enter the numeric features expected by your trained model. "
    "V1–V28 are anonymized features from the public dataset."
)

with st.form("transaction_form"):
    values = {}

    columns = st.columns(3)

    for i, feature in enumerate(feature_columns):
        with columns[i % 3]:
            values[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f",
            )

    submitted = st.form_submit_button(
        "Assess transaction",
        type="primary",
        use_container_width=True,
    )

if submitted:
    row = pd.DataFrame(
        [[values[c] for c in feature_columns]],
        columns=feature_columns,
    )

    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(row)[0, 1])
    else:
        score = float(model.predict(row)[0])

    prediction = int(score >= threshold)

    left, right = st.columns(2)

    with left:
        st.metric("Model score", f"{score:.4f}")

    with right:
        st.metric("Decision threshold", f"{threshold:.4f}")

    if prediction == 1:
        st.error("⚠️ Fraud-risk flag")
        st.write("The model score meets or exceeds the configured threshold.")
    else:
        st.success("✓ Not flagged by the model")
        st.write("The model score is below the configured threshold.")

    st.caption(
        "Zero-valued inputs are synthetic. A model score is not necessarily "
        "a calibrated real-world fraud probability."
    )
    st.divider()
st.header("Batch Fraud Detection")

uploaded_file = st.file_uploader(
    "Upload sample_transactions.csv",
    type=["csv"]
)

if uploaded_file is not None:
    transactions = pd.read_csv(uploaded_file)

    missing = set(feature_columns) - set(transactions.columns)

    if missing:
        st.error(f"Missing columns: {sorted(missing)}")
    else:
        transactions = transactions[feature_columns]

        scores = model.predict_proba(transactions)[:, 1]
        predictions = (scores >= threshold).astype(int)

        results = transactions.copy()
        results["Fraud Score"] = scores
        results["Prediction"] = [
            "Fraud-risk flag" if p == 1 else "Not flagged"
            for p in predictions
        ]

        st.dataframe(results, use_container_width=True)

        st.download_button(
            "Download Predictions",
            results.to_csv(index=False),
            "fraud_predictions.csv",
            "text/csv"
        )

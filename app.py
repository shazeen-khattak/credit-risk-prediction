"""
app.py

Interactive Streamlit dashboard for the Credit Risk Prediction model.

Lets a user enter loan applicant details and see:
- The model's predicted probability of default
- A risk category (Low / Medium / High)
- A SHAP-based explanation of which factors drove that prediction

Run locally with:
    streamlit run app.py

Requires a trained model and scaler to be saved to disk first
(see save_model_artifacts() in src/model_training.py, or train and
pickle them directly - see the "Model Artifacts" section of the
project README for the exact commands).
"""

import pickle

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk Predictor", layout="wide")


@st.cache_resource
def load_artifacts():
    """Load the trained model and the list of feature columns it expects."""
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model("model_artifacts/xgb_model.json")

    with open("model_artifacts/feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    return model, feature_columns


def build_input_row(feature_columns: list, user_inputs: dict) -> pd.DataFrame:
    """
    Build a single-row dataframe matching the model's expected
    feature columns, filling anything not exposed in the UI with a
    reasonable default (0), and applying the same categorical
    encodings used during training for the fields the UI exposes.
    """
    row = {col: 0 for col in feature_columns}

    row["loan_amnt"] = user_inputs["loan_amnt"]
    row["term"] = 1 if user_inputs["term"] == "60 months" else 0
    row["int_rate"] = user_inputs["int_rate"]
    row["installment"] = user_inputs["loan_amnt"] * (user_inputs["int_rate"] / 1200) / (
        1 - (1 + user_inputs["int_rate"] / 1200) ** (-36 if user_inputs["term"] == "36 months" else -60)
    )
    row["annual_inc"] = user_inputs["annual_inc"]
    row["dti"] = user_inputs["dti"]

    grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
    row["grade"] = grade_map[user_inputs["grade"]]
    row["sub_grade"] = row["grade"] * 5  # approximate mid sub-grade

    emp_length_map = {
        "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
        "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
        "10+ years": 10,
    }
    row["emp_length"] = emp_length_map[user_inputs["emp_length"]]

    row["credit_history_years"] = user_inputs["credit_history_years"]
    row["loan_to_income_ratio"] = min(
        user_inputs["loan_amnt"] / (user_inputs["annual_inc"] + 1), 0.5
    )
    row["revol_util"] = user_inputs["revol_util"]
    row["open_acc"] = user_inputs["open_acc"]

    home_col = f"home_ownership_{user_inputs['home_ownership']}"
    if home_col in row:
        row[home_col] = 1

    return pd.DataFrame([row])[feature_columns]


def categorize_risk(probability: float) -> tuple:
    """Map a predicted probability to a risk label and color."""
    if probability < 0.15:
        return "Low Risk", "green"
    elif probability < 0.35:
        return "Medium Risk", "orange"
    else:
        return "High Risk", "red"


def main():
    st.title("Credit Risk Prediction Dashboard")
    st.markdown(
        "Enter loan applicant details to estimate the probability of default, "
        "based on an XGBoost model trained on the Lending Club dataset (2007-2015)."
    )

    try:
        model, feature_columns = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Model artifacts not found. Please run the training pipeline and save "
            "the model first (see README.md, 'Model Artifacts' section)."
        )
        return

    st.sidebar.header("Applicant & Loan Details")

    loan_amnt = st.sidebar.number_input("Loan Amount ($)", min_value=1000, max_value=40000, value=15000, step=500)
    term = st.sidebar.selectbox("Term", ["36 months", "60 months"])
    int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 31.0, 13.5, step=0.1)
    grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    annual_inc = st.sidebar.number_input("Annual Income ($)", min_value=0, max_value=500000, value=65000, step=1000)
    emp_length = st.sidebar.selectbox(
        "Employment Length",
        ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
         "6 years", "7 years", "8 years", "9 years", "10+ years"],
        index=5,
    )
    home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN"])
    dti = st.sidebar.slider("Debt-to-Income Ratio (%)", 0.0, 50.0, 18.0, step=0.5)
    revol_util = st.sidebar.slider("Revolving Credit Utilization (%)", 0.0, 150.0, 45.0, step=1.0)
    open_acc = st.sidebar.number_input("Open Credit Accounts", min_value=0, max_value=50, value=10)
    credit_history_years = st.sidebar.slider("Credit History Length (years)", 1.0, 40.0, 12.0, step=0.5)

    user_inputs = {
        "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate, "grade": grade,
        "annual_inc": annual_inc, "emp_length": emp_length, "home_ownership": home_ownership,
        "dti": dti, "revol_util": revol_util, "open_acc": open_acc,
        "credit_history_years": credit_history_years,
    }

    input_row = build_input_row(feature_columns, user_inputs)
    probability = model.predict_proba(input_row)[0, 1]
    risk_label, risk_color = categorize_risk(probability)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prediction")
        st.metric("Predicted Default Probability", f"{probability * 100:.1f}%")
        st.markdown(f"### Risk Category: :{risk_color}[{risk_label}]")

    with col2:
        st.subheader("Why this prediction?")
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(input_row)

            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)
        except Exception as e:
            st.info(f"SHAP explanation unavailable: {e}")

    st.markdown("---")
    st.caption(
        "This tool is for portfolio/demonstration purposes only and should not be "
        "used to make real lending decisions."
    )


if __name__ == "__main__":
    main()

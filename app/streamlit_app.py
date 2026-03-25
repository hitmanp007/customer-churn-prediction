import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
st.sidebar.title("📊 Customer Churn App")

page = st.sidebar.radio("Navigation", [
    "Prediction",
    "Dashboard",
    "Model Insights"
])

# Load model
BASE_DIR = Path(__file__).resolve().parent.parent

model = pickle.load(open(BASE_DIR / "model" / "churn_model.pkl", "rb"))
model_columns = pickle.load(open(BASE_DIR / "model" / "model_columns.pkl", "rb"))

st.title("📊 Customer Churn Prediction")

st.write("Enter customer details to predict churn risk")

# User inputs
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 2000.0)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox("Online Security", ["Yes", "No"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

# Create input dataframe
import numpy as np

# Create empty input with ALL columns
input_df = pd.DataFrame(
    np.zeros((1, len(model_columns))),
    columns=model_columns
)

# Fill numeric values
input_df["tenure"] = tenure
input_df["MonthlyCharges"] = monthly_charges
input_df["TotalCharges"] = total_charges

# Fill categorical EXACTLY
input_df[f"Contract_{contract}"] = 1
input_df[f"InternetService_{internet_service}"] = 1
input_df[f"OnlineSecurity_{online_security}"] = 1
input_df[f"TechSupport_{tech_support}"] = 1
input_df[f"PaymentMethod_{payment_method}"] = 1
input_df[f"PaperlessBilling_{paperless_billing}"] = 1

# Align columns with training columns
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[model_columns]

# Prediction
if st.button("Predict Churn"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if probability > 0.3:
        st.error(f"⚠ Customer likely to churn. Probability: {probability:.2f}")
    else:
        st.success(f"✅ Customer likely to stay. Probability: {probability:.2f}")
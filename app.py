import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("../model/churn_model.pkl", "rb"))
model_columns = pickle.load(open("../model/model_columns.pkl", "rb"))

st.title("Customer Churn Prediction App")

st.write("Enter customer details to predict churn risk.")

# User Inputs
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)

contract = st.selectbox("Contract Type",
                        ["Month-to-month", "One year", "Two year"])

internet_service = st.selectbox("Internet Service",
                                ["DSL", "Fiber optic", "No"])

tech_support = st.selectbox("Tech Support", ["Yes", "No"])

online_security = st.selectbox("Online Security", ["Yes", "No"])

payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

# Convert input to dataframe
input_data = pd.DataFrame({
    "tenure":[tenure],
    "MonthlyCharges":[monthly_charges],
})

# Predict button
if st.button("Predict Churn"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠ Customer likely to churn. Probability: {probability:.2f}")
    else:
        st.success(f"✅ Customer likely to stay. Probability: {probability:.2f}")
# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
with open("loan_approval_model.pkl", "rb") as f:
    model, scaler, encoders = pickle.load(f)

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered",
)

# ------------------- CUSTOM CSS -------------------
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 32px;
            font-weight: 700;
            color: #1a73e8;
        }
        .sub-title {
            text-align: center;
            font-size: 16px;
            color: #555;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------- TITLE -------------------
st.markdown('<p class="main-title">🏦 Loan Approval Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Enter applicant details to check loan eligibility.</p>', unsafe_allow_html=True)
st.write("---")

# ------------------- USER INPUT FORM -------------------
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0, step=100)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=100)
        loan_amount = st.number_input("Loan Amount", min_value=0, step=10)
        loan_amount_term = st.selectbox("Loan Amount Term (Months)", [12, 36, 60, 120, 180, 240, 300, 360, 480])
        credit_history = st.selectbox("Credit History (1 = Good, 0 = Bad)", [0, 1])

    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submit_btn = st.form_submit_button("🔍 Predict")

# ------------------- DATA PROCESS -------------------
input_dict = {
    "Gender": [gender],
    "Married": [married],
    "Dependents": [dependents],
    "Education": [education],
    "Self_Employed": [self_employed],
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_amount_term],
    "Credit_History": [credit_history],
    "Property_Area": [property_area],
}

df_input = pd.DataFrame(input_dict)

for col in df_input.columns:
    if col in encoders:
        df_input[col] = encoders[col].transform(df_input[col].astype(str))

df_scaled = scaler.transform(df_input)

# ------------------- PREDICTION -------------------
if submit_btn:
    result = model.predict(df_scaled)[0]
    if result == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Rejected")

# ------------------- SIDEBAR -------------------
st.sidebar.title("👤 Developer Info")
st.sidebar.markdown("**Tejas Gholap**")
st.sidebar.write("MCA Student | ML & AI Enthusiast")

st.sidebar.markdown("[🔗 GitHub](https://github.com/tejasgholap45)")
st.sidebar.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/tejas-gholap-bb3417300/)")

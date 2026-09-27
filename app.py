import streamlit as st
import pandas as pd
import joblib

# Load trained models
logistic_model = joblib.load("logistic_model.pkl")
linear_model = joblib.load("linear_model.pkl")

# Page configuration
st.set_page_config(
    page_title="ABC Ltd - AI Employee Analytics",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("ABC Ltd")
st.subheader("AI-Powered Employee Analytics & Decision Support Tool")

st.write(
    "This tool uses predictive analytics to support managerial "
    "decision-making related to employee attrition and income."
)

st.info(
    "Important: These predictions are intended as decision-support "
    "information and should not be used as the sole basis for "
    "managerial decisions."
)

# ==========================================
# 1. EMPLOYEE ATTRITION PREDICTION
# ==========================================

st.header("1. Employee Attrition Prediction")

st.write(
    "Enter employee information to estimate the probability "
    "of employee attrition."
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=70,
    value=30
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=200000,
    value=50000
)

job_level = st.selectbox(
    "Job Level",
    [1, 2, 3, 4, 5]
)

job_satisfaction = st.selectbox(
    "Job Satisfaction",
    [1, 2, 3, 4]
)

years_at_company = st.number_input(
    "Years at Company",
    min_value=0,
    max_value=50,
    value=3
)

overtime = st.selectbox(
    "Overtime",
    ["Yes", "No"]
)

work_life_balance = st.selectbox(
    "Work-Life Balance",
    [1, 2, 3, 4]
)

distance_from_home = st.number_input(
    "Distance From Home",
    min_value=1,
    max_value=100,
    value=5
)

total_working_years = st.number_input(
    "Total Working Years",
    min_value=0,
    max_value=50,
    value=5
)

if st.button("Predict Employee Attrition"):

    attrition_input = pd.DataFrame({
        "Age": [age],
        "MonthlyIncome": [monthly_income],
        "JobLevel": [job_level],
        "JobSatisfaction": [job_satisfaction],
        "YearsAtCompany": [years_at_company],
        "OverTime": [overtime],
        "WorkLifeBalance": [work_life_balance],
        "DistanceFromHome": [distance_from_home],
        "TotalWorkingYears": [total_working_years]
    })

    probability = logistic_model.predict_proba(
        attrition_input
    )[0][1]

    st.subheader("AI Prediction")

    st.metric(
        "Predicted Attrition Probability",
        f"{probability:.1%}"
    )

    if probability < 0.30:
        risk = "Low"
    elif probability < 0.60:
        risk = "Medium"
    else:
        risk = "High"

    st.write(f"### Predicted Risk Level: {risk}")


# ==========================================
# 2. MONTHLY INCOME PREDICTION
# ==========================================

st.divider()

st.header("2. Monthly Income Prediction")

st.write(
    "Enter employee characteristics to estimate monthly income."
)

income_age = st.number_input(
    "Age for Income Prediction",
    min_value=18,
    max_value=70,
    value=30,
    key="income_age"
)

income_job_level = st.selectbox(
    "Job Level for Income Prediction",
    [1, 2, 3, 4, 5],
    key="income_job_level"
)

income_total_years = st.number_input(
    "Total Working Years for Income Prediction",
    min_value=0,
    max_value=50,
    value=5,
    key="income_total_years"
)

income_years_company = st.number_input(
    "Years at Company for Income Prediction",
    min_value=0,
    max_value=50,
    value=3,
    key="income_years_company"
)

years_current_role = st.number_input(
    "Years in Current Role",
    min_value=0,
    max_value=30,
    value=2
)

years_since_promotion = st.number_input(
    "Years Since Last Promotion",
    min_value=0,
    max_value=30,
    value=1
)

performance_rating = st.selectbox(
    "Performance Rating",
    [1, 2, 3, 4],
    index=2
)

if st.button("Predict Monthly Income"):

    income_input = pd.DataFrame({
        "Age": [income_age],
        "JobLevel": [income_job_level],
        "TotalWorkingYears": [income_total_years],
        "YearsAtCompany": [income_years_company],
        "YearsInCurrentRole": [years_current_role],
        "YearsSinceLastPromotion": [years_since_promotion],
        "PerformanceRating": [performance_rating]
    })

    predicted_income = linear_model.predict(
        income_input
    )[0]

    st.subheader("AI Prediction")

    st.metric(
        "Predicted Monthly Income",
        f"{predicted_income:,.0f}"
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "ABC Ltd | Predictive Analytics & Managerial AI Adoption Study"
)

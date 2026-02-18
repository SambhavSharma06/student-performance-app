import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------

st.set_page_config(
    page_title="Student Performance Analytics",
    layout="wide"
)

# ------------------------------------------------
# Load Dataset
# ------------------------------------------------

df = pd.read_csv("The_Real_Student_Performance.csv")

# Load trained model and scaler
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    ["Project Overview", "Data Exploration", "Machine Learning Models"]
)

# ==========================================================
# PAGE 1 — PROJECT OVERVIEW
# ==========================================================

if page == "Project Overview":

    st.title("Student Performance Data Analytics Project")

    st.subheader("Project Objective")
    st.write("""
    This project aims to analyse student performance data and predict
    final grades using machine learning models. The goal is to understand
    the factors that influence academic success.
    """)

    st.subheader("Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Students", df.shape[0])
        st.metric("Total Features", df.shape[1])

    with col2:
        st.metric("Target Variable", "final_grade")
        st.metric("Type of Problem", "Classification")

    st.subheader("Approach Used")

    st.write("""
    1. Exploratory Data Analysis (EDA)
    2. Data Cleaning and Preprocessing
    3. Encoding Categorical Variables
    4. Feature Scaling
    5. Model Training (Logistic Regression, Decision Tree, Random Forest)
    6. Model Evaluation and Comparison
    """)

    st.subheader("Why This Matters")

    st.write("""
    Understanding student performance patterns helps identify
    important academic factors such as study hours, attendance,
    and subject scores. Machine learning allows us to make
    data-driven predictions.
    """)

# ==========================================================
# PAGE 2 — DATA EXPLORATION
# ==========================================================

elif page == "Data Exploration":

    st.title("Exploratory Data Analysis")

    st.subheader("Apply Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.multiselect(
            "Select Gender",
            df["gender"].unique()
        )

    with col2:
        school = st.multiselect(
            "Select School Type",
            df["school_type"].unique()
        )

    with col3:
        parent_edu = st.multiselect(
            "Select Parent Education",
            df["parent_education"].unique()
        )

    filtered_df = df.copy()

    if gender:
        filtered_df = filtered_df[filtered_df["gender"].isin(gender)]

    if school:
        filtered_df = filtered_df[filtered_df["school_type"].isin(school)]

    if parent_edu:
        filtered_df = filtered_df[filtered_df["parent_education"].isin(parent_edu)]

    st.subheader("Filtered Dataset")
    st.dataframe(filtered_df)

    st.subheader("Study Hours vs Overall Score")
    st.scatter_chart(filtered_df[["study_hours", "overall_score"]])

    st.subheader("Attendance vs Final Grade")
    st.bar_chart(filtered_df.groupby("final_grade")["attendance_percentage"].mean())

# ==========================================================
# PAGE 3 — MACHINE LEARNING MODELS
# ==========================================================

elif page == "Machine Learning Models":

    st.title("Machine Learning Model Comparison")

    st.subheader("Models Used")

    st.write("""
    - Logistic Regression (Baseline Model)
    - Decision Tree
    - Random Forest
    """)

    # Replace these with your real accuracy values
    log_acc = 0.79
    dt_acc = 0.84
    rf_acc = 0.87

    st.subheader("Model Accuracy Comparison")

    model_results = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Accuracy": [log_acc, dt_acc, rf_acc]
    })

    st.dataframe(model_results)
    st.bar_chart(model_results.set_index("Model"))

    st.subheader("Best Performing Model")

    st.write("""
    Based on evaluation results, Random Forest achieved the highest accuracy.
    Therefore, it was selected as the final model for prediction.
    """)

    st.divider()

    st.subheader("Predict Final Grade")

    col1, col2, col3 = st.columns(3)

    with col1:
        study_hours = st.slider("Study Hours", 0.0, 10.0, 5.0)

    with col2:
        attendance = st.slider("Attendance Percentage", 0, 100, 75)

    with col3:
        math_score = st.slider("Math Score", 0, 100, 60)

    if st.button("Predict Grade"):

        input_data = np.array([[study_hours, attendance, math_score]])

        scaled_input = scaler.transform(input_data)
        prediction = rf_model.predict(scaled_input)

        st.success(f"Predicted Final Grade: {prediction[0]}")

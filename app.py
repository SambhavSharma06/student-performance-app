import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide"
)

# ------------------------------------------------
# Load Data
# ------------------------------------------------

df = pd.read_csv("The_Real_Student_Performance.csv")
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "",
    ["🏠 Project Overview", "📊 Data Exploration", "🤖 Machine Learning Models"]
)

# ==========================================================
# PAGE 1 — PROJECT OVERVIEW
# ==========================================================

if page == "🏠 Project Overview":

    st.title("🎓 Student Performance Data Analytics Project")

    st.markdown("""
    ### Project Introduction

    This project focuses on analysing student academic performance using
    data analytics and machine learning techniques.

    The main objective is to identify the key factors that influence
    student success and to build predictive models capable of forecasting
    final grades.
    """)

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", df.shape[0])
    col2.metric("Total Features", df.shape[1])
    col3.metric("Prediction Type", "Classification")

    st.divider()

    st.subheader("📌 Methodology Followed")

    st.markdown("""
    The project was completed using the following structured approach:

    1. **Exploratory Data Analysis (EDA)**  
       Understanding data distribution, relationships, and patterns.

    2. **Data Preprocessing**  
       Cleaning missing values, encoding categorical variables, and scaling features.

    3. **Feature Engineering**  
       Selecting relevant features influencing academic performance.

    4. **Model Training**  
       Logistic Regression, Decision Tree, and Random Forest were implemented.

    5. **Model Evaluation**  
       Models were compared using accuracy, precision, recall, and confusion matrices.
    """)

    st.divider()

    st.subheader("🎯 Why This Project Matters")

    st.markdown("""
    Education analytics allows institutions to better understand
    performance trends and identify students who may need additional support.

    By applying machine learning techniques, predictions can be made
    early, enabling proactive academic intervention strategies.
    """)

# ==========================================================
# PAGE 2 — DATA EXPLORATION
# ==========================================================

elif page == "📊 Data Exploration":

    st.title("📊 Exploratory Data Analysis")

    st.markdown("""
    This section allows interactive exploration of the dataset.
    Filters can be applied to analyse how different factors impact performance.
    """)

    st.divider()

    st.subheader("🔎 Apply Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.multiselect(
            "Gender",
            df["gender"].unique()
        )

    with col2:
        school = st.multiselect(
            "School Type",
            df["school_type"].unique()
        )

    with col3:
        parent_edu = st.multiselect(
            "Parent Education Level",
            df["parent_education"].unique()
        )

    col4, col5 = st.columns(2)

    with col4:
        internet_access = st.multiselect(
            "Internet Access",
            df["internet_access"].unique()
        )

    with col5:
        extra_activities = st.multiselect(
            "Extra Activities",
            df["extra_activities"].unique()
        )

    # Apply filters
    filtered_df = df.copy()

    if gender:
        filtered_df = filtered_df[filtered_df["gender"].isin(gender)]

    if school:
        filtered_df = filtered_df[filtered_df["school_type"].isin(school)]

    if parent_edu:
        filtered_df = filtered_df[filtered_df["parent_education"].isin(parent_edu)]

    if internet_access:
        filtered_df = filtered_df[filtered_df["internet_access"].isin(internet_access)]

    if extra_activities:
        filtered_df = filtered_df[filtered_df["extra_activities"].isin(extra_activities)]

    st.divider()

    st.subheader("📋 Filtered Dataset Overview")

    st.metric("Filtered Students", filtered_df.shape[0])
    st.dataframe(filtered_df)

    st.divider()

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Study Hours vs Overall Score")
        st.scatter_chart(filtered_df[["study_hours", "overall_score"]])

    with colB:
        st.subheader("Attendance by Final Grade")
        st.bar_chart(filtered_df.groupby("final_grade")["attendance_percentage"].mean())

# ==========================================================
# PAGE 3 — MACHINE LEARNING MODELS
# ==========================================================

elif page == "🤖 Machine Learning Models":

    st.title("🤖 Machine Learning Model Comparison")

    st.markdown("""
    Three classification algorithms were implemented to predict final grades.
    The performance of each model was evaluated and compared.
    """)

    # Replace with your real values
    log_acc = 0.79
    dt_acc = 0.84
    rf_acc = 0.87

    model_results = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Accuracy": [log_acc, dt_acc, rf_acc]
    })

    st.subheader("📈 Model Performance Comparison")
    st.dataframe(model_results)
    st.bar_chart(model_results.set_index("Model"))

    st.divider()

    st.subheader("🏆 Best Performing Model")

    st.markdown("""
    Random Forest achieved the highest overall accuracy,
    indicating better generalisation and prediction capability.

    Therefore, it was selected as the final deployed model.
    """)

    st.divider()

    st.subheader("🔮 Predict Student Final Grade")

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

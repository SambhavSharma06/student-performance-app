import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and scaler
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset
df = pd.read_csv("The_Real_Student_Performance.csv")

st.set_page_config(page_title="Student Performance Analytics", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Project Overview", "Data Exploration", "Machine Learning Models"]
)

# ---------------------------------------------------------
# PAGE 1 — PROJECT OVERVIEW
# ---------------------------------------------------------

if page == "Project Overview":

    st.title("Student Performance Analytics Project")

    st.markdown("""
    This project analyses student academic performance using data analytics and machine learning.
    
    The objective is to understand which factors influence final grades and build predictive models 
    that can estimate student outcomes.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", df.shape[0])
    col2.metric("Total Features", df.shape[1])
    col3.metric("Target Variable", "final_grade")

    st.markdown("""
    ### Approach Used

    • Exploratory Data Analysis (EDA)  
    • Data Preprocessing (Encoding + Scaling)  
    • Model Training (Logistic Regression, Decision Tree, Random Forest)  
    • Model Evaluation (Accuracy, Precision, Recall, F1-score)  

    Random Forest achieved the highest performance and was selected as the final model.
    """)

    st.markdown("### Sample Data")
    st.dataframe(df.head())

# ---------------------------------------------------------
# PAGE 2 — DATA EXPLORATION
# ---------------------------------------------------------

elif page == "Data Exploration":

    st.title("Exploratory Data Analysis")

    st.markdown("Use the filters below to explore the dataset dynamically.")

    col1, col2 = st.columns(2)

    gender_filter = col1.multiselect(
        "Select Gender",
        options=df["gender"].unique(),
        default=df["gender"].unique()
    )

    school_filter = col2.multiselect(
        "Select School Type",
        options=df["school_type"].unique(),
        default=df["school_type"].unique()
    )

    parent_filter = st.multiselect(
        "Select Parent Education Level",
        options=df["parent_education"].unique(),
        default=df["parent_education"].unique()
    )

    filtered_df = df[
        (df["gender"].isin(gender_filter)) &
        (df["school_type"].isin(school_filter)) &
        (df["parent_education"].isin(parent_filter))
    ]

    st.markdown("### Filtered Dataset")
    st.dataframe(filtered_df)

    st.markdown("### Study Hours vs Final Grade")
    st.scatter_chart(filtered_df[["study_hours", "overall_score"]])

    st.markdown("### Attendance vs Final Grade")
    st.scatter_chart(filtered_df[["attendance_percentage", "overall_score"]])

# ---------------------------------------------------------
# PAGE 3 — MACHINE LEARNING MODELS
# ---------------------------------------------------------

elif page == "Machine Learning Models":

    st.title("Machine Learning Model Prediction")

    st.markdown("""
    The Random Forest model was selected as the best-performing algorithm 
    based on accuracy and balanced classification metrics.
    """)

    st.subheader("Enter Student Details")

    col1, col2, col3 = st.columns(3)

    study_hours = col1.slider("Study Hours", 0, 10, 5)
    attendance = col2.slider("Attendance Percentage", 0, 100, 75)
    math_score = col3.slider("Math Score", 0, 100, 60)

    science_score = col1.slider("Science Score", 0, 100, 60)
    english_score = col2.slider("English Score", 0, 100, 60)

    gender = col1.selectbox("Gender", df["gender"].unique())
    school_type = col2.selectbox("School Type", df["school_type"].unique())
    parent_education = col3.selectbox("Parent Education", df["parent_education"].unique())
    internet_access = col1.selectbox("Internet Access", df["internet_access"].unique())
    extra_activities = col2.selectbox("Extra Activities", df["extra_activities"].unique())

    if st.button("Predict Final Grade"):

        input_dict = {
            "study_hours": study_hours,
            "attendance_percentage": attendance,
            "math_score": math_score,
            "science_score": science_score,
            "english_score": english_score,
            "gender": gender,
            "school_type": school_type,
            "parent_education": parent_education,
            "internet_access": internet_access,
            "extra_activities": extra_activities
        }

        input_df = pd.DataFrame([input_dict])

        # Apply same encoding as training
        input_df = pd.get_dummies(input_df)

        # Align with training features
        input_df = input_df.reindex(
            columns=rf_model.feature_names_in_,
            fill_value=0
        )

        # Scale input
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = rf_model.predict(scaled_input)

        st.success(f"Predicted Final Grade: {prediction[0]}")

        st.markdown("""
        This prediction is generated using the trained Random Forest model,
        which was selected as the final model due to superior performance.
        """)

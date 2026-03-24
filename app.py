# -----------------------------------------------------
# IMPORT LIBRARIES
# -----------------------------------------------------

import streamlit as st                      # For building web app
import pandas as pd                         # For handling data
import numpy as np                          # For numerical operations
import joblib                               # For loading saved model

# -----------------------------------------------------
# PAGE SETTINGS
# -----------------------------------------------------

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("The_Real_Student_Performance.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# -----------------------------------------------------
# REMOVE USELESS COLUMNS
# -----------------------------------------------------

if "student_id" in df.columns:
    df = df.drop("student_id", axis=1)

if "overall_score" in df.columns:
    df = df.drop("overall_score", axis=1)

# -----------------------------------------------------
# TARGET VARIABLE
# -----------------------------------------------------

target_column = "final_grade"

# -----------------------------------------------------
# PREPARE FEATURES (ONLY FOR STRUCTURE)
# -----------------------------------------------------

X = df.drop(target_column, axis=1)

# Apply dummy encoding
X = pd.get_dummies(X)

# Save structure
feature_columns = X.columns

# -----------------------------------------------------
# LOAD TRAINED MODEL AND SCALER (IMPORTANT)
# -----------------------------------------------------

rf_model = joblib.load("rf_model.pkl")   # Load trained Random Forest model
scaler = joblib.load("scaler.pkl")       # Load scaler

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Project Overview",
        "Dataset Exploration",
        "Machine Learning Models",
        "Prediction System"
    ]
)

# -----------------------------------------------------
# PAGE 1 — OVERVIEW
# -----------------------------------------------------

if page == "Project Overview":

    st.title("🎓 Student Performance Analytics System")

    st.write("""
This project analyzes student academic performance using machine learning.

The system studies how factors such as study hours, attendance,
school type, and student background influence academic results.

The model was trained using multiple algorithms, and Random Forest
achieved the highest accuracy and was selected as the final model.
""")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Model Used", "Random Forest")

# -----------------------------------------------------
# PAGE 2 — DATA EXPLORATION
# -----------------------------------------------------

elif page == "Dataset Exploration":

    st.title("📊 Dataset Exploration")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Distribution of Final Grades")
    grade_counts = df["final_grade"].value_counts()
    st.bar_chart(grade_counts)

# -----------------------------------------------------
# PAGE 3 — MODEL COMPARISON
# -----------------------------------------------------

elif page == "Machine Learning Models":

    st.title("🤖 Machine Learning Model Comparison")

    st.subheader("Accuracy Score (All Models)")

    st.code("""
Logistic Regression Accuracy: 0.7632
Decision Tree Accuracy: 0.8600
Random Forest Accuracy: 0.9024
""")

    st.success("Random Forest achieved the highest accuracy and was selected as the final model.")

    model_data = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Accuracy": [0.7632, 0.86, 0.9024]
    })

    st.subheader("Model Accuracy Comparison")
    st.bar_chart(model_data.set_index("Model"))

# -----------------------------------------------------
# PAGE 4 — PREDICTION SYSTEM
# -----------------------------------------------------

elif page == "Prediction System":

    st.title("🎯 Predict Student Final Grade")

    st.write("""
Enter student information below.

The trained Random Forest model will predict
the student's final grade.
""")

    input_data = {}

    for col in df.columns:

        if col == target_column:
            continue

        if df[col].dtype == "object":

            input_data[col] = st.selectbox(
                col.replace("_"," ").title(),
                df[col].unique()
            )

        else:

            input_data[col] = st.slider(
                col.replace("_"," ").title(),
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean())
            )

    if st.button("Predict Grade"):

        input_df = pd.DataFrame([input_data])

        # Apply same dummy encoding
        input_df = pd.get_dummies(input_df)

        # Match training structure
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Apply scaling
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = rf_model.predict(input_scaled)[0]

        st.success(f"Predicted Final Grade: {prediction}")

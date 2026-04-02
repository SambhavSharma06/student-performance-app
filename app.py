# ==============================
# IMPORT LIBRARIES
# ==============================

import streamlit as st              # Used to build the web app
import pandas as pd                 # Used to handle dataset (tables)
import numpy as np                  # Used for numerical operations
from joblib import load             # Used to load saved model files


# ==============================
# PAGE SETTINGS
# ==============================

st.set_page_config(
    page_title="Student Performance Analytics",   # Title in browser tab
    page_icon="🎓",                               # Icon in browser tab
    layout="wide"                                 # Full-width layout
)


# ==============================
# LOAD DATASET
# ==============================

@st.cache_data                               # Cache data → faster loading
def load_data():
    df = pd.read_csv("The_Real_Student_Performance.csv")   # Load CSV file
    df.columns = df.columns.str.strip()                   # Remove extra spaces in column names
    df = df.dropna()                                     # Remove missing values (important)
    return df

df = load_data()                                          # Call function


# ==============================
# LOAD MODEL + SCALER + COLUMNS
# ==============================

@st.cache_resource                           # Cache model (loads once only)
def load_model():
    model = load("rf_model.joblib")          # Load trained Random Forest model
    scaler = load("scaler.joblib")           # Load scaler used during training
    columns = load("columns.joblib")         # Load exact training feature columns
    return model, scaler, columns

rf_model, scaler, feature_columns = load_model()   # Get all components


# ==============================
# CLEAN DATA
# ==============================

# Remove useless columns if they exist
if "student_id" in df.columns:
    df = df.drop("student_id", axis=1)

if "overall_score" in df.columns:
    df = df.drop("overall_score", axis=1)

target_column = "final_grade"   # This is what we want to predict


# ==============================
# SIDEBAR NAVIGATION
# ==============================

st.sidebar.title("📊 Navigation")   # Sidebar title

page = st.sidebar.radio(
    "Select Section",
    [
        "Project Overview",
        "Dataset Exploration",
        "Machine Learning Models",
        "Prediction System"
    ]
)


# ==============================
# PAGE 1 — PROJECT OVERVIEW
# ==============================

if page == "Project Overview":

    st.title("🎓 Student Performance Analytics System")

    st.write("""
This project predicts student final grades using machine learning.

It analyzes factors like study hours, attendance, and background.
""")

    # Show metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))   # Number of rows
    col2.metric("Total Columns", 16)         # Fixed number of features
    col3.metric("Models Used", 3)            # Total ML models used


# ==============================
# PAGE 2 — DATASET EXPLORATION
# ==============================

elif page == "Dataset Exploration":

    st.title("📊 Dataset Exploration")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())   # Show first 5 rows

    st.subheader("Statistical Summary")
    st.write(df.describe())   # Show mean, std, etc.

    st.subheader("Final Grade Distribution")
    st.bar_chart(df["final_grade"].value_counts())   # Show grade distribution


# ==============================
# PAGE 3 — MACHINE LEARNING MODELS
# ==============================

elif page == "Machine Learning Models":

    st.title("🤖 Machine Learning Models")

    st.write("""
This project compares three machine learning models:
""")

    # Describe models
    st.write("""
1. Logistic Regression → Basic model  
2. Decision Tree → Rule-based model  
3. Random Forest → Best performing model  
""")

    # Create performance table
    model_data = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Accuracy": [0.76, 0.86, 0.90]
    })

    # Show chart
    st.bar_chart(model_data.set_index("Model"))

    # Highlight best model
    st.success("🏆 Best Model: Random Forest")


# ==============================
# PAGE 4 — PREDICTION SYSTEM
# ==============================

elif page == "Prediction System":

    st.title("🎯 Predict Student Final Grade")

    input_data = {}   # Dictionary to store user inputs

    # Loop through each column
    for i, col in enumerate(df.columns):

        if col == target_column:
            continue   # Skip target column

        unique_key = f"{col}_{i}"   # Unique key for Streamlit widgets

        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):

            input_data[col] = st.slider(
                col.replace("_", " ").title(),   # Label
                float(df[col].min()),            # Min value
                float(df[col].max()),            # Max value
                float(df[col].mean()),           # Default value
                key=unique_key
            )

        else:
            # For categorical data → dropdown
            input_data[col] = st.selectbox(
                col.replace("_", " ").title(),
                df[col].astype(str).unique(),
                key=unique_key
            )


    # ==========================
    # PREDICT BUTTON
    # ==========================

    if st.button("Predict Grade"):

        try:
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])

            # Convert categorical → dummy variables
            input_df = pd.get_dummies(input_df)

            # Match training columns exactly
            input_df = input_df.reindex(columns=feature_columns, fill_value=0)

            # Scale input
            input_scaled = scaler.transform(input_df)

            # Predict using model
            prediction = rf_model.predict(input_scaled)[0]

            # Show result
            st.success(f"🎯 Predicted Final Grade: {prediction}")

        except Exception as e:
            # Show error if something fails
            st.error(f"Error: {e}")

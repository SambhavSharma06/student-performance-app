# ===============================
# IMPORT LIBRARIES
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# ===============================
# PAGE SETTINGS
# ===============================

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide"
)

# ===============================
# LOAD DATA
# ===============================

@st.cache_data
def load_data():
    df = pd.read_csv("The_Real_Student_Performance.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ===============================
# LOAD MODEL + SCALER
# ===============================

@st.cache_resource
def load_model():
    model = load("rf_model.joblib")
    scaler = load("scaler.joblib")
    return model, scaler

rf_model, scaler = load_model()

# ===============================
# PREPROCESSING
# ===============================

# Drop unnecessary columns
if "student_id" in df.columns:
    df = df.drop("student_id", axis=1)

if "overall_score" in df.columns:
    df = df.drop("overall_score", axis=1)

target_column = "final_grade"

X = df.drop(target_column, axis=1)
y = df[target_column]

# Create dummy variables (same as training)
X = pd.get_dummies(X)

# Save feature structure
feature_columns = X.columns

# ===============================
# SIDEBAR NAVIGATION
# ===============================

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

# ===============================
# PAGE 1 — OVERVIEW
# ===============================

if page == "Project Overview":

    st.title("🎓 Student Performance Analytics System")

    st.write("""
This project predicts student final grades using machine learning.
""")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Models Used", 3)

# ===============================
# PAGE 2 — DATASET
# ===============================

elif page == "Dataset Exploration":

    st.title("📊 Dataset Exploration")

    st.dataframe(df.head())
    st.write(df.describe())

# ===============================
# PAGE 3 — MODELS
# ===============================

elif page == "Machine Learning Models":

    st.title("🤖 Model Performance")

    st.code("""
Logistic Regression: 0.76
Decision Tree: 0.86
Random Forest: 0.90
""")

    st.success("Random Forest performed best.")

# ===============================
# PAGE 4 — PREDICTION (FIXED)
# ===============================

elif page == "Prediction System":

    st.title("🎯 Predict Student Final Grade")

    input_data = {}

    # 🔥 Use ORIGINAL columns (NOT dummy columns)
    for col in df.columns:

        if col == target_column:
            continue

        if df[col].dtype == "object":

            input_data[col] = st.selectbox(
                col,
                df[col].unique(),
                key=col   # 🔥 prevents duplicate error
            )

        else:

            input_data[col] = st.slider(
                col,
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean()),
                key=col   # 🔥 prevents duplicate error
            )

    # ===============================
    # PREDICT BUTTON
    # ===============================

    if st.button("Predict Grade"):

        input_df = pd.DataFrame([input_data])

        # Apply same dummy encoding
        input_df = pd.get_dummies(input_df)

        # 🔥 Match training columns exactly
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = rf_model.predict(input_scaled)[0]

        st.success(f"Predicted Final Grade: {prediction}")

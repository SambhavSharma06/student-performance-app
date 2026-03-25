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
# PREPROCESS (IMPORTANT)
# ===============================

# Remove unnecessary columns
if "student_id" in df.columns:
    df = df.drop("student_id", axis=1)

if "overall_score" in df.columns:
    df = df.drop("overall_score", axis=1)

target_column = "final_grade"

X = df.drop(target_column, axis=1)
y = df[target_column]

# Create dummy columns (IMPORTANT)
X = pd.get_dummies(X)

# Save training feature structure
feature_columns = X.columns

# ===============================
# SIDEBAR
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
# PAGE 1
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
# PAGE 2
# ===============================

elif page == "Dataset Exploration":

    st.title("📊 Dataset Exploration")

    st.dataframe(df.head())
    st.write(df.describe())

# ===============================
# PAGE 3
# ===============================

elif page == "Machine Learning Models":

    st.title("🤖 Model Performance")

    st.code("""
Logistic Regression: 0.76
Decision Tree: 0.86
Random Forest: 0.90
""")

    st.success("Random Forest is best.")

# ===============================
# PAGE 4 (FIXED)
# ===============================

elif page == "Prediction System":

    st.title("🎯 Predict Student Final Grade")

    input_data = {}

    for col in X.columns:

        # Skip dummy columns (we will rebuild them)
        original_col = col.split("_")[0]

        if original_col not in df.columns or original_col == target_column:
            continue

        if df[original_col].dtype == "object":

            input_data[original_col] = st.selectbox(
                original_col,
                df[original_col].unique()
            )

        else:

            input_data[original_col] = st.slider(
                original_col,
                float(df[original_col].min()),
                float(df[original_col].max()),
                float(df[original_col].mean())
            )

    if st.button("Predict"):

        input_df = pd.DataFrame([input_data])

        # APPLY DUMMIES (CRITICAL)
        input_df = pd.get_dummies(input_df)

        # MATCH TRAINING COLUMNS (THIS FIXES ERROR)
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # SCALE
        input_scaled = scaler.transform(input_df)

        # PREDICT
        prediction = rf_model.predict(input_scaled)[0]

        st.success(f"Predicted Grade: {prediction}")

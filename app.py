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
# PREPROCESSING (MATCH TRAINING)
# ===============================
if "student_id" in df.columns:
    df = df.drop("student_id", axis=1)

if "overall_score" in df.columns:
    df = df.drop("overall_score", axis=1)

target_column = "final_grade"

X = df.drop(target_column, axis=1)

# Convert categorical → numeric
X = pd.get_dummies(X)
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
# PAGE 1 — OVERVIEW
# ===============================
if page == "Project Overview":

    st.title("🎓 Student Performance Analytics")

    st.write("""
This app predicts student final grades using Machine Learning.

Best Model: Random Forest
""")

    col1, col2, col3 = st.columns(3)

    col1.metric("Students", len(df))
    col2.metric("Features", 16)
    col3.metric("Model", 3)


# ===============================
# PAGE 2 — DATA
# ===============================
elif page == "Dataset Exploration":

    st.title("📊 Dataset")

    st.dataframe(df.head())

    st.subheader("Statistics")
    st.write(df.describe())

    st.subheader("Grade Distribution")
    st.bar_chart(df["final_grade"].value_counts())


# ===============================
# PAGE 3 — MODELS
# ===============================
elif page == "Machine Learning Models":

    st.title("🤖 Model Results")

    st.write("""
Logistic Regression: 0.76  
Decision Tree: 0.86  
Random Forest: 0.90 ✅
""")

    st.success("Random Forest is best")

    chart = pd.DataFrame({
        "Model": ["LR", "DT", "RF"],
        "Accuracy": [0.76, 0.86, 0.90]
    })

    st.bar_chart(chart.set_index("Model"))


# ===============================
# PAGE 4 — PREDICTION
# ===============================
elif page == "Prediction System":

    st.title("🎯 Predict Grade")

    input_data = {}

    for col in df.columns:

        if col == target_column:
            continue

        if df[col].dtype == "object":
            input_data[col] = st.selectbox(col, df[col].unique())
        else:
            input_data[col] = st.slider(
                col,
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean())
            )

    if st.button("Predict"):

        input_df = pd.DataFrame([input_data])

        # Same encoding as training
        input_df = pd.get_dummies(input_df)

        # Match columns
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = rf_model.predict(input_scaled)[0]

        st.success(f"Predicted Grade: {prediction}")

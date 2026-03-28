import streamlit as st
import pandas as pd
import numpy as np
from joblib import load


st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide"
)


@st.cache_data
def load_data():
    df = pd.read_csv("The_Real_Student_Performance.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

@st.cache_resource
def load_model():
    model = load("rf_model.joblib")
    scaler = load("scaler.joblib")
    columns = load("columns.joblib")
    return model, scaler, columns

rf_model, scaler, feature_columns = load_model()

if "student_id" in df.columns:
    df = df.drop("student_id", axis=1)

if "overall_score" in df.columns:
    df = df.drop("overall_score", axis=1)

target_column = "final_grade"


st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Project Overview",
        "Dataset Exploration",
        "Machine Learning Models",   # ⭐ NEW PAGE
        "Prediction System"
    ]
)


if page == "Project Overview":

    st.title("🎓 Student Performance Analytics System")

    st.write("""
This project predicts student final grades using machine learning.

It considers factors like study hours, attendance, and background.
""")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Total Columns", 16)   # ⭐ FIXED
    col3.metric("Models Used", 3)      # ⭐ NEW

elif page == "Dataset Exploration":

    st.title("📊 Dataset Exploration")

    st.dataframe(df.head())
    st.write(df.describe())

    st.subheader("Final Grade Distribution")
    st.bar_chart(df["final_grade"].value_counts())


elif page == "Machine Learning Models":

    st.title("🤖 Machine Learning Models Used")

    st.write("""
This project compares three machine learning models to predict student performance.
""")


    st.subheader("📌 Models Used")

    st.write("""
1. Logistic Regression – Simple baseline model  
2. Decision Tree – Captures non-linear patterns  
3. Random Forest – Ensemble model (Best Performance)
""")
=
    st.subheader("📊 Model Performance")

    model_data = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Accuracy": [0.76, 0.86, 0.90]
    })

    st.bar_chart(model_data.set_index("Model"))


    st.success("🏆 Best Model: Random Forest (Highest Accuracy)")


elif page == "Prediction System":

    st.title("🎯 Predict Student Final Grade")

    input_data = {}

    for i, col in enumerate(df.columns):

        if col == target_column:
            continue

        unique_key = f"{col}_{i}"

        if df[col].dtype == "object":

            input_data[col] = st.selectbox(
                col.replace("_", " ").title(),
                df[col].unique(),
                key=unique_key
            )

        else:

            input_data[col] = st.slider(
                col.replace("_", " ").title(),
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean()),
                key=unique_key
            )

    if st.button("Predict Grade"):

        try:
            input_df = pd.DataFrame([input_data])
            input_df = pd.get_dummies(input_df)

       
            input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        
            input_scaled = scaler.transform(input_df)

 
            prediction = rf_model.predict(input_scaled)[0]

            st.success(f"🎯 Predicted Final Grade: {prediction}")

        except Exception as e:
            st.error(f"Error: {e}")

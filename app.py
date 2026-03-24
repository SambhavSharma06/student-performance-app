

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os  


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



if "student_id" in df.columns:
    df = df.drop("student_id", axis=1)

if "overall_score" in df.columns:
    df = df.drop("overall_score", axis=1)



target_column = "final_grade"


X = df.drop(target_column, axis=1)
X = pd.get_dummies(X)
feature_columns = X.columns



if os.path.exists("rf_model.pkl") and os.path.exists("scaler.pkl"):

    rf_model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")

else:
    st.error("Model files not found. Please upload rf_model.pkl and scaler.pkl")
    st.stop()



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


if page == "Project Overview":

    st.title("🎓 Student Performance Analytics System")

    st.write("""
This project uses machine learning to predict student final grades.

The model was trained using student data such as study hours,
attendance, and academic background.

Random Forest achieved the highest accuracy and was selected
as the final model.
""")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Model Used", "Random Forest")


elif page == "Dataset Exploration":

    st.title("📊 Dataset Exploration")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Distribution of Final Grades")
    grade_counts = df["final_grade"].value_counts()
    st.bar_chart(grade_counts)

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


elif page == "Prediction System":

    st.title("🎯 Predict Student Final Grade")

    st.write("""
Enter student details below to predict the final grade.
""")

    input_data = {}

    for col in df.columns:

        if col == target_column:
            continue

        if df[col].dtype == "object":

            input_data[col] = st.selectbox(
                col.replace("_", " ").title(),
                df[col].unique()
            )

        else:

            input_data[col] = st.slider(
                col.replace("_", " ").title(),
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean())
            )

    if st.button("Predict Grade"):

        input_df = pd.DataFrame([input_data])

        # Apply same encoding
        input_df = pd.get_dummies(input_df)

        # Match training columns
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = rf_model.predict(input_scaled)[0]

        st.success(f"Predicted Final Grade: {prediction}")

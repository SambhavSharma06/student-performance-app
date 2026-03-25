# ===============================
# IMPORT LIBRARIES
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import pickle


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
# LOAD MODEL + SCALER (IMPORTANT)
# ===============================
@st.cache_resource
def load_model():
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

rf_model, scaler = load_model()


# ===============================
# PREPROCESSING (SAME AS TRAINING)
# ===============================
if "student_id" in df.columns:
    df = df.drop("student_id", axis=1)

if "overall_score" in df.columns:
    df = df.drop("overall_score", axis=1)

target_column = "final_grade"

X = df.drop(target_column, axis=1)
y = df[target_column]

# Dummy encoding
X = pd.get_dummies(X)
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
This project predicts student final grades using Machine Learning.

It analyzes:
- Study habits
- Attendance
- Academic scores

Best Model: Random Forest 🌲
""")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Total Features", len(df.columns))
    col3.metric("Models Used", 3)


# ===============================
# PAGE 2 — DATA EXPLORATION
# ===============================
elif page == "Dataset Exploration":

    st.title("📊 Dataset Exploration")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Final Grade Distribution")
    st.bar_chart(df["final_grade"].value_counts())


# ===============================
# PAGE 3 — MODEL INFO
# ===============================
elif page == "Machine Learning Models":

    st.title("🤖 Model Comparison")

    st.write("""
Logistic Regression Accuracy: 0.76  
Decision Tree Accuracy: 0.86  
Random Forest Accuracy: 0.90 ✅
""")

    st.success("Random Forest is the best model")

    chart_data = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Accuracy": [0.76, 0.86, 0.90]
    })

    st.bar_chart(chart_data.set_index("Model"))


# ===============================
# PAGE 4 — PREDICTION SYSTEM
# ===============================
elif page == "Prediction System":

    st.title("🎯 Predict Student Final Grade")

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

        # Dummy encoding
        input_df = pd.get_dummies(input_df)

        # Match training columns
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = rf_model.predict(input_scaled)[0]

        st.success(f"🎉 Predicted Final Grade: {prediction}")

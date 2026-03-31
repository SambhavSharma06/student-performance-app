# importing all required libraries
import streamlit as st          # for building the web app UI
import pandas as pd            # for handling data
import numpy as np             # for numerical operations
from joblib import load        # to load saved model files


# setting up the page (title, icon, layout)
st.set_page_config(
    page_title="Student Performance Analytics",   # title shown on browser tab
    page_icon="🎓",                                # icon for the app
    layout="wide"                                 # makes layout wide instead of narrow
)


# caching the dataset so it doesn't reload every time
@st.cache_data
def load_data():
    df = pd.read_csv("The_Real_Student_Performance.csv")   # loading dataset
    df.columns = df.columns.str.strip()                   # removing extra spaces from column names
    return df                                             # returning cleaned dataset


# calling the function to actually load data
df = load_data()


# caching model loading so it doesn't reload again and again
@st.cache_resource
def load_model():
    model = load("rf_model.joblib")       # loading trained Random Forest model
    scaler = load("scaler.joblib")        # loading scaler used during training
    columns = load("columns.joblib")      # loading column structure (important!)
    return model, scaler, columns         # returning all three


# storing loaded model, scaler, and columns
rf_model, scaler, feature_columns = load_model()


# removing student_id if present (not useful for prediction)
if "student_id" in df.columns:
    df = df.drop("student_id", axis=1)


# removing overall_score because it leaks information (too direct)
if "overall_score" in df.columns:
    df = df.drop("overall_score", axis=1)


# defining target column (what we want to predict)
target_column = "final_grade"


# sidebar navigation title
st.sidebar.title("📊 Navigation")


# creating navigation options
page = st.sidebar.radio(
    "Select Section",
    [
        "Project Overview",
        "Dataset Exploration",
        "Machine Learning Models",  
        "Prediction System"
    ]
)


# PROJECT OVERVIEW PAGE
if page == "Project Overview":

    st.title("🎓 Student Performance Analytics System")

    # simple explanation of project
    st.write("""
This project predicts student final grades using machine learning.

It considers factors like study hours, attendance, and background.
""")

    # creating 3 columns for metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))   # number of rows
    col2.metric("Total Columns", 16)         # total features (fixed)
    col3.metric("Models Used", 3)            # number of ML models


# DATASET EXPLORATION PAGE
elif page == "Dataset Exploration":

    st.title("📊 Dataset Exploration")

    st.dataframe(df.head())      # showing first few rows of data
    st.write(df.describe())      # showing statistical summary

    st.subheader("Final Grade Distribution")
    st.bar_chart(df["final_grade"].value_counts())   # visualizing class distribution


# MACHINE LEARNING MODELS PAGE 
elif page == "Machine Learning Models":

    st.title("🤖 Machine Learning Models Used")

    st.write("""
This project compares three machine learning models to predict student performance.
""")

    st.subheader("📌 Models Used")

    # explaining models in simple way
    st.write("""
1. Logistic Regression – Simple baseline model  
2. Decision Tree – Captures non-linear patterns  
3. Random Forest – Ensemble model (Best Performance)
""")

  
    st.subheader("📊 Model Performance")

    # manually creating performance table
    model_data = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Accuracy": [0.76, 0.86, 0.90]   # accuracy scores
    })

    st.bar_chart(model_data.set_index("Model"))   # plotting performance

    # highlighting best model
    st.success("🏆 Best Model: Random Forest (Highest Accuracy)")


# PREDICTION SYSTEM PAGE
elif page == "Prediction System":

    st.title("🎯 Predict Student Final Grade")

    input_data = {}   # dictionary to store user inputs

    # looping through all columns to create input fields
    for i, col in enumerate(df.columns):

        if col == target_column:
            continue   # skip target column

        unique_key = f"{col}_{i}"   # unique key for Streamlit widgets

        # if column is categorical (text)
        if df[col].dtype == "object":

            input_data[col] = st.selectbox(
                col.replace("_", " ").title(),   # clean display name
                df[col].unique(),               # options from dataset
                key=unique_key
            )

        # if column is numerical
        else:

            input_data[col] = st.slider(
                col.replace("_", " ").title(),   # clean label
                float(df[col].min()),            # min value
                float(df[col].max()),            # max value
                float(df[col].mean()),           # default value
                key=unique_key
            )


    # when user clicks predict button
    if st.button("Predict Grade"):

        try:
            input_df = pd.DataFrame([input_data])   # convert input to dataframe

            input_df = pd.get_dummies(input_df)     # encode categorical values

            # match training columns (very important)
            input_df = input_df.reindex(columns=feature_columns, fill_value=0)

            input_scaled = scaler.transform(input_df)   # scale input data

            prediction = rf_model.predict(input_scaled)[0]   # make prediction

            st.success(f"🎯 Predicted Final Grade: {prediction}")   # show result

        except Exception as e:
            st.error(f"Error: {e}")   # show error if something goes wrong

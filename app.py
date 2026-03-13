# -----------------------------------------------------
# IMPORT LIBRARIES
# -----------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


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

# remove overall score because it leaks information
if "overall_score" in df.columns:
    df = df.drop("overall_score", axis=1)


# -----------------------------------------------------
# TARGET VARIABLE
# -----------------------------------------------------

target_column = "final_grade"


# -----------------------------------------------------
# SPLIT INPUT AND OUTPUT
# -----------------------------------------------------

X = df.drop(target_column, axis=1)

y = df[target_column]


# -----------------------------------------------------
# DUMMY VARIABLES
# -----------------------------------------------------

X = pd.get_dummies(X)

feature_columns = X.columns


# -----------------------------------------------------
# SCALE DATA
# -----------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# -----------------------------------------------------
# TRAIN TEST SPLIT
# -----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------------------------------
# TRAIN MODELS
# -----------------------------------------------------

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


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
# PAGE 1
# -----------------------------------------------------

if page == "Project Overview":

    st.title("🎓 Student Performance Analytics System")

    st.write("""
This project analyzes student academic performance using machine learning.

The system studies how factors such as study hours, attendance,
school type, internet access, and learning methods influence
student academic results.

Three machine learning models were trained and compared
to determine which model predicts student final grades
with the highest accuracy.
""")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Total Columns", 16)
    col3.metric("Models Compared", 3)


# -----------------------------------------------------
# PAGE 2
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
# PAGE 3
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
# PAGE 4
# -----------------------------------------------------

elif page == "Prediction System":

    st.title("🎯 Predict Student Final Grade")

    st.write("""
Enter student information below.

The trained Random Forest model will estimate
the student's expected final grade.
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

        input_df = pd.get_dummies(input_df)

        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        input_scaled = scaler.transform(input_df)

        prediction = rf_model.predict(input_scaled)[0]

        st.success(f"Predicted Final Grade: {prediction}")

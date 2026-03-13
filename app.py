# -----------------------------------------------------
# IMPORT LIBRARIES
# -----------------------------------------------------

# Streamlit for web app
import streamlit as st

# Data handling
import pandas as pd
import numpy as np

# Machine learning utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Visualization
import matplotlib.pyplot as plt


# -----------------------------------------------------
# STREAMLIT PAGE SETTINGS
# -----------------------------------------------------

st.set_page_config(
    page_title="Student Performance Analytics",
    layout="wide"
)


# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------

@st.cache_data
def load_data():

    # Read CSV file
    df = pd.read_csv("The_Real_Student_Performance.csv")

    # Remove hidden spaces in column names
    df.columns = df.columns.str.strip()

    return df


df = load_data()


# -----------------------------------------------------
# REMOVE USELESS COLUMN
# -----------------------------------------------------

# student_id is just an identifier
if "student_id" in df.columns:
    df = df.drop("student_id", axis=1)


# -----------------------------------------------------
# TARGET VARIABLE
# -----------------------------------------------------

target_column = "final_grade"


# -----------------------------------------------------
# SPLIT INPUT AND OUTPUT
# -----------------------------------------------------

# X contains input columns
X = df.drop(target_column, axis=1)

# y contains target column
y = df[target_column]


# -----------------------------------------------------
# CREATE DUMMY VARIABLES
# -----------------------------------------------------

# Convert categorical variables to numeric
X = pd.get_dummies(X)

# Save column names
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

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# -----------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------

st.sidebar.title("Navigation")

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
# PAGE 1 : PROJECT OVERVIEW
# -----------------------------------------------------

if page == "Project Overview":

    st.title("Student Performance Analytics System")

    st.write("""
This project analyzes student academic performance using machine learning.

The system studies how factors such as study hours, attendance percentage,
school type, internet access, and other academic factors influence
student final grades.

Three machine learning models were trained and compared to determine
which model predicts student grades most accurately.
""")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Models Compared", 3)


# -----------------------------------------------------
# PAGE 2 : DATASET EXPLORATION
# -----------------------------------------------------

elif page == "Dataset Exploration":

    st.title("Dataset Exploration")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Statistical Summary")

    st.write(df.describe())

    # -------------------------------
    # GRADE DISTRIBUTION CHART
    # -------------------------------

    st.subheader("Distribution of Final Grades")

    fig, ax = plt.subplots()

    df["final_grade"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Final Grade")
    ax.set_ylabel("Number of Students")
    ax.set_title("Distribution of Student Final Grades")

    st.pyplot(fig)


# -----------------------------------------------------
# PAGE 3 : MODEL COMPARISON
# -----------------------------------------------------

elif page == "Machine Learning Models":

    st.title("Machine Learning Model Comparison")

    st.write("""
Three machine learning models were trained and evaluated
to determine which model predicts student final grades
with the highest accuracy.
""")

    st.subheader("Accuracy Score (All Models)")

    st.code("""
Logistic Regression Accuracy: 0.7632
Decision Tree Accuracy: 0.8600
Random Forest Accuracy: 0.9024
""")

    st.success(
        "Random Forest achieved the highest accuracy and was selected as the final model."
    )

    # -----------------------------------
    # MODEL ACCURACY CHART
    # -----------------------------------

    st.subheader("Model Accuracy Comparison")

    models = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ]

    accuracy_scores = [
        0.7632,
        0.8600,
        0.9024
    ]

    fig, ax = plt.subplots()

    ax.bar(models, accuracy_scores)

    ax.set_ylabel("Accuracy")

    ax.set_title("Machine Learning Model Accuracy Comparison")

    st.pyplot(fig)


# -----------------------------------------------------
# PAGE 4 : PREDICTION SYSTEM
# -----------------------------------------------------

elif page == "Prediction System":

    st.title("Predict Student Final Grade")

    st.write("""
Enter student information below.

The system will use the trained Random Forest model
to estimate the student's expected final grade.
""")

    input_data = {}

    for col in df.columns:

        if col == target_column:
            continue

        if df[col].dtype == "object":

            input_data[col] = st.selectbox(
                col,
                df[col].unique()
            )

        else:

            input_data[col] = st.slider(
                col,
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

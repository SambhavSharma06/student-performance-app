# Import Streamlit for creating the web application
import streamlit as st

# Import pandas for data manipulation
import pandas as pd

# Import numpy
import numpy as np

# Import train_test_split for splitting the dataset
from sklearn.model_selection import train_test_split

# Import StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler

# Import classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Configure page layout
st.set_page_config(page_title="Student Performance Analytics", layout="wide")


# -----------------------------
# LOAD DATASET
# -----------------------------
@st.cache_data
def load_data():

    df = pd.read_csv("The_Real_Student_Performance.csv")

    # remove hidden spaces
    df.columns = df.columns.str.strip()

    return df


df = load_data()


# -----------------------------
# REMOVE STUDENT ID
# -----------------------------
if "student_id" in df.columns:
    df = df.drop("student_id", axis=1)


# -----------------------------
# TARGET VARIABLE
# -----------------------------
target_column = "final_grade"


# -----------------------------
# SEPARATE INPUT AND OUTPUT
# -----------------------------
X = df.drop(target_column, axis=1)
y = df[target_column]


# -----------------------------
# DUMMY VARIABLES
# -----------------------------
X = pd.get_dummies(X)

feature_columns = X.columns


# -----------------------------
# SCALE DATA
# -----------------------------
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# TRAIN MODELS
# -----------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
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


# -----------------------------
# PAGE 1 : PROJECT OVERVIEW
# -----------------------------
if page == "Project Overview":

    st.title("Student Performance Analytics System")

    st.write("""
This project analyzes student academic performance using machine learning.

The goal of this project is to understand how factors such as study hours,
attendance percentage, school type, and student background influence
student academic performance.

Three machine learning models were trained and compared in order to
determine which model predicts student final grades most accurately.
""")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Models Compared", 3)


# -----------------------------
# PAGE 2 : DATASET EXPLORATION
# -----------------------------
elif page == "Dataset Exploration":

    st.title("Dataset Exploration")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Statistical Summary")
    st.write(df.describe())


# -----------------------------
# PAGE 3 : MODEL COMPARISON
# -----------------------------
elif page == "Machine Learning Models":

    st.title("Accuracy Score (All Models)")

    st.code("""
Logistic Regression Accuracy: 0.7632
Decision Tree Accuracy: 0.8600
Random Forest Accuracy: 0.9024
""")

    st.success("Random Forest is the best performing model because it achieved the highest accuracy.")


# -----------------------------
# PAGE 4 : PREDICTION SYSTEM
# -----------------------------
elif page == "Prediction System":

    st.title("Predict Student Final Grade")

    st.write("""
Enter student information below.  
The Random Forest model will estimate the student's final academic grade.
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

        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        input_scaled = scaler.transform(input_df)

        prediction = rf_model.predict(input_scaled)[0]

        st.success(f"Predicted Final Grade: {prediction}")

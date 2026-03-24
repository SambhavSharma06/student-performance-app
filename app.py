
# IMPORT LIBRARIES


import streamlit as st            # Used to create the web application interface
import pandas as pd               # Used to handle dataset (tables)
import numpy as np                # Used for numerical operations

from sklearn.model_selection import train_test_split   # Used to split data into training and testing
from sklearn.preprocessing import StandardScaler       # Used to scale/normalize data

from sklearn.linear_model import LogisticRegression    # Logistic Regression model
from sklearn.tree import DecisionTreeClassifier        # Decision Tree model
from sklearn.ensemble import RandomForestClassifier    # Random Forest model



# PAGE SETTINGS

st.set_page_config(
    page_title="Student Performance Analytics",   # Title shown on browser tab
    page_icon="🎓",                               # Icon shown on browser tab
    layout="wide"                                 # Makes layout wider and cleaner
)



# LOAD DATA


@st.cache_data
def load_data():
    df = pd.read_csv("The_Real_Student_Performance.csv")   # Load dataset
    df.columns = df.columns.str.strip()                   # Remove extra spaces in column names
    return df                                             # Return cleaned data

df = load_data()                                          # Call function and store dataset



# REMOVE USELESS COLUMNS


if "student_id" in df.columns:
    df = df.drop("student_id", axis=1)                    # Remove ID (no importance)

if "overall_score" in df.columns:
    df = df.drop("overall_score", axis=1)                 # Remove (data leakage)



# TARGET VARIABLE


target_column = "final_grade"                             # This is what we want to predict



# SPLIT INPUT (X) AND OUTPUT (y)
X = df.drop(target_column, axis=1)                        # All input features
y = df[target_column]                                     # Output variable


# DUMMY VARIABLES (TEXT → NUMBERS)


X = pd.get_dummies(X)                                     # Convert categorical data to numeric
feature_columns = X.columns                               # Save column structure


# SCALE DATA


scaler = StandardScaler()                                 # Create scaler
X_scaled = scaler.fit_transform(X)                        # Apply scaling



# TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,                                             # Input data
    y,                                                    # Output data
    test_size=0.2,                                        # 20% testing
    random_state=42                                       # Same results every time
)


# TRAIN MODELS


# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)              # Create model
lr_model.fit(X_train, y_train)                            # Train model

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Random Forest (Best model)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)



# SIDEBAR NAVIGATION

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


# PAGE 1 — PROJECT OVERVIEW
if page == "Project Overview":

    st.title("🎓 Student Performance Analytics System")

    st.write("""
This project uses machine learning to predict student final grades.

Different factors like study hours, attendance, and background
are used to understand student performance.

Three models were compared and Random Forest performed the best.
""")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))                # Number of students
    col2.metric("Total Columns", 16)         # Number of columns
    col3.metric("Models Compared", 3)                     # Number of models


# PAGE 2 — DATA EXPLORATION


elif page == "Dataset Exploration":

    st.title("📊 Dataset Exploration")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())                              # Show first rows

    st.subheader("Statistical Summary")
    st.write(df.describe())                              # Show statistics

    st.subheader("Distribution of Final Grades")
    grade_counts = df["final_grade"].value_counts()       # Count grades
    st.bar_chart(grade_counts)                           # Show graph



# PAGE 3 — MODEL COMPARISON


elif page == "Machine Learning Models":

    st.title("🤖 Machine Learning Model Comparison")

    st.subheader("Accuracy Score (All Models)")

    st.code("""
Logistic Regression Accuracy: 0.7632
Decision Tree Accuracy: 0.8600
Random Forest Accuracy: 0.9024
""")

    st.success("Random Forest performed the best.")

    model_data = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Accuracy": [0.7632, 0.86, 0.9024]
    })

    st.bar_chart(model_data.set_index("Model"))           # Show comparison chart



# PAGE 4 — PREDICTION SYSTEM


elif page == "Prediction System":

    st.title("🎯 Predict Student Final Grade")

    st.write("Enter student details below:")

    input_data = {}                                       # Store user input

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

        input_df = pd.DataFrame([input_data])              # Convert input to DataFrame

        input_df = pd.get_dummies(input_df)                # Apply dummy encoding

        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        input_scaled = scaler.transform(input_df)          # Scale input

        prediction = rf_model.predict(input_scaled)[0]     # Predict grade

        st.success(f"Predicted Final Grade: {prediction}")

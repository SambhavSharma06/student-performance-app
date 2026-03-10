# Import Streamlit for building the web app
import streamlit as st

# Import pandas for data handling
import pandas as pd

# Import numpy
import numpy as np

# Import train_test_split to split data
from sklearn.model_selection import train_test_split

# Import StandardScaler for scaling data
from sklearn.preprocessing import StandardScaler

# Import machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Import accuracy metric
from sklearn.metrics import accuracy_score


# Configure Streamlit page
st.set_page_config(page_title="Student Performance Analytics", layout="wide")


# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------

@st.cache_data
def load_data():

    # Read dataset
    df = pd.read_csv("The_Real_Student_Performance.csv")

    # Remove hidden spaces
    df.columns = df.columns.str.strip()

    return df


# Load dataset
df = load_data()


# ----------------------------------------------------
# REMOVE USELESS COLUMN
# ----------------------------------------------------

# student_id is only an identifier
if "student_id" in df.columns:
    df = df.drop("student_id", axis=1)


# ----------------------------------------------------
# TARGET VARIABLE
# ----------------------------------------------------

# Target variable is final_grade
target_column = "final_grade"


# ----------------------------------------------------
# SEPARATE INPUT AND OUTPUT
# ----------------------------------------------------

# X contains input columns
X = df.drop(target_column, axis=1)

# y contains target column
y = df[target_column]


# ----------------------------------------------------
# CREATE DUMMY VARIABLES
# ----------------------------------------------------

# Convert categorical columns into numeric
X = pd.get_dummies(X)

# Save column names
feature_columns = X.columns


# ----------------------------------------------------
# SCALE DATA
# ----------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ----------------------------------------------------
# SPLIT DATASET
# ----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)


# ----------------------------------------------------
# TRAIN MODELS
# ----------------------------------------------------

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Decision Tree
dt_model = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)
dt_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)


# ----------------------------------------------------
# MODEL EVALUATION
# ----------------------------------------------------

# Logistic Regression accuracy
lr_pred = lr_model.predict(X_test)
lr_score = accuracy_score(y_test, lr_pred)

# Decision Tree accuracy
dt_pred = dt_model.predict(X_test)
dt_score = accuracy_score(y_test, dt_pred)

# Random Forest accuracy
rf_pred = rf_model.predict(X_test)
rf_score = accuracy_score(y_test, rf_pred)


# ----------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------

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


# ----------------------------------------------------
# PAGE 1 : PROJECT OVERVIEW
# ----------------------------------------------------

if page == "Project Overview":

    st.title("Student Performance Analytics System")

    st.write("""
This project analyzes student academic performance using machine learning.

Different factors such as study hours, attendance, school type,
internet access, and learning methods are used to understand
how students perform academically.

Three machine learning models are trained and compared
to determine which model predicts final grades most accurately.
""")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Models Compared", 3)


# ----------------------------------------------------
# PAGE 2 : DATASET EXPLORATION
# ----------------------------------------------------

elif page == "Dataset Exploration":

    st.title("Dataset Exploration")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Statistical Summary")

    st.write(df.describe())


# ----------------------------------------------------
# PAGE 3 : MACHINE LEARNING MODELS
# ----------------------------------------------------

elif page == "Machine Learning Models":

    st.title("Machine Learning Model Comparison")

    st.write("""
Three machine learning models were trained and evaluated
to determine which model predicts student final grades
with the highest accuracy.
""")

    st.subheader("Accuracy Score (All Models)")

    st.code(f"""
Logistic Regression Accuracy: {lr_score:.4f}
Decision Tree Accuracy: {dt_score:.4f}
Random Forest Accuracy: {rf_score:.4f}
""")

    scores = {

        "Logistic Regression": lr_score,
        "Decision Tree": dt_score,
        "Random Forest": rf_score
    }

    best_model = max(scores, key=scores.get)

    st.success(f"The best performing model is: {best_model}")


# ----------------------------------------------------
# PAGE 4 : PREDICTION SYSTEM
# ----------------------------------------------------

elif page == "Prediction System":

    st.title("Predict Student Final Grade")

    st.write("""
Enter the student information below.

The system will use the trained Random Forest model
to estimate the student's final academic grade.
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

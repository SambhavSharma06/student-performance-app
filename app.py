# Import Streamlit to create the web application interface
import streamlit as st

# Import pandas to read and manipulate data
import pandas as pd

# Import numpy for numerical operations
import numpy as np

# Import function to split dataset
from sklearn.model_selection import train_test_split

# Import StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler

# Import classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Import evaluation metric
from sklearn.metrics import accuracy_score


# Configure Streamlit page
st.set_page_config(page_title="Student Performance Analytics", layout="wide")


# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

@st.cache_data
def load_data():

    # Read dataset
    df = pd.read_csv("The_Real_Student_Performance.csv")

    # Remove hidden spaces from column names
    df.columns = df.columns.str.strip()

    return df


# Load dataset
df = load_data()


# ------------------------------------------------
# REMOVE USELESS COLUMN
# ------------------------------------------------

# student_id does not affect predictions
if "student_id" in df.columns:
    df = df.drop("student_id", axis=1)


# Target variable we want to predict
target_column = "final_grade"


# ------------------------------------------------
# CREATE DUMMY VARIABLES
# ------------------------------------------------

# Convert categorical columns into numerical dummy variables
df_encoded = pd.get_dummies(df)


# Separate input features from target
X = df_encoded.drop(target_column, axis=1)
y = df_encoded[target_column]


# Store feature names
feature_columns = X.columns


# ------------------------------------------------
# FEATURE SCALING
# ------------------------------------------------

# Create scaler
scaler = StandardScaler()

# Scale input features
X_scaled = scaler.fit_transform(X)


# ------------------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------------------

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)


# ------------------------------------------------
# TRAIN MODELS
# ------------------------------------------------

# Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)

# Train Logistic Regression
lr_model.fit(X_train, y_train)


# Decision Tree model
dt_model = DecisionTreeClassifier(

    max_depth=10,
    random_state=42
)

# Train Decision Tree
dt_model.fit(X_train, y_train)


# Random Forest model
rf_model = RandomForestClassifier(

    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Train Random Forest
rf_model.fit(X_train, y_train)


# ------------------------------------------------
# MODEL EVALUATION
# ------------------------------------------------

# Logistic Regression predictions
lr_pred = lr_model.predict(X_test)

# Logistic Regression accuracy
lr_score = accuracy_score(y_test, lr_pred)


# Decision Tree predictions
dt_pred = dt_model.predict(X_test)

# Decision Tree accuracy
dt_score = accuracy_score(y_test, dt_pred)


# Random Forest predictions
rf_pred = rf_model.predict(X_test)

# Random Forest accuracy
rf_score = accuracy_score(y_test, rf_pred)


# ------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------

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


# ------------------------------------------------
# PAGE 1 : PROJECT OVERVIEW
# ------------------------------------------------

if page == "Project Overview":

    st.title("Student Performance Analytics System")

    st.write("""
This project predicts student final grades using machine learning.

Different factors such as study hours, attendance,
school type, internet access, and academic performance
are used to understand how students perform academically.

Three machine learning models are trained and compared
to determine which model predicts final grades most accurately.
""")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Number of Features", len(feature_columns))
    col3.metric("Models Compared", 3)


# ------------------------------------------------
# PAGE 2 : DATASET EXPLORATION
# ------------------------------------------------

elif page == "Dataset Exploration":

    st.title("Dataset Exploration")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Statistical Summary")

    st.write(df.describe())


# ------------------------------------------------
# PAGE 3 : MODEL COMPARISON
# ------------------------------------------------

elif page == "Machine Learning Models":

    st.title("Machine Learning Model Comparison")

    col1, col2, col3 = st.columns(3)

    col1.metric("Logistic Regression Accuracy", round(lr_score,3))
    col2.metric("Decision Tree Accuracy", round(dt_score,3))
    col3.metric("Random Forest Accuracy", round(rf_score,3))

    scores = {
        "Logistic Regression": lr_score,
        "Decision Tree": dt_score,
        "Random Forest": rf_score
    }

    best_model = max(scores, key=scores.get)

    st.success(f"Best Performing Model: {best_model}")


# ------------------------------------------------
# PAGE 4 : PREDICTION SYSTEM
# ------------------------------------------------

elif page == "Prediction System":

    st.title("Predict Student Final Grade")

    st.write("""
Enter student information below.

The machine learning model will estimate
the student's expected final grade.
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

        input_df = input_df.reindex(columns=X.columns, fill_value=0)

        input_scaled = scaler.transform(input_df)

        prediction = rf_model.predict(input_scaled)[0]

        st.success(f"Predicted Final Grade: {prediction}")

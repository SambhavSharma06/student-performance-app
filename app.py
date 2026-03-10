# Import Streamlit library to create the web application interface
import streamlit as st

# Import pandas for reading and manipulating datasets
import pandas as pd

# Import numpy for numerical operations
import numpy as np

# Import train_test_split to divide dataset into training and testing data
from sklearn.model_selection import train_test_split

# Import StandardScaler to scale numerical values
from sklearn.preprocessing import StandardScaler

# Import three machine learning models
from sklearn.linear_model import LinearRegression        # Linear Regression model
from sklearn.tree import DecisionTreeRegressor           # Decision Tree model
from sklearn.ensemble import RandomForestRegressor       # Random Forest model

# Import evaluation metric to measure model performance
from sklearn.metrics import r2_score


# Configure the Streamlit page
# page_title sets the browser tab name
# layout="wide" allows full screen width for better layout
st.set_page_config(page_title="Student Performance Analytics", layout="wide")

# LOAD DATA

# Cache the dataset loading function so the file is not read repeatedly
@st.cache_data
def load_data():

    # Read the dataset from CSV file
    df = pd.read_csv("The_Real_Student_Performance.csv")

    # Remove hidden spaces from column names
    df.columns = df.columns.str.strip()

    # Return the cleaned dataframe
    return df


# Call the function to load the dataset
df = load_data()

# REMOVE USELESS COLUMNS

# student_id is just an identifier and does not affect predictions
# final_grade is removed to avoid data leakage
drop_columns = ["student_id", "final_grade"]

# Loop through the columns to safely remove them
for col in drop_columns:

    # Check if the column exists in the dataset
    if col in df.columns:

        # Remove the column from the dataset
        df = df.drop(col, axis=1)


# Define the target variable that we want to predict
target_column = "overall_score"

# CONVERT CATEGORICAL DATA USING DUMMY VARIABLES

# Convert categorical columns into dummy variables
# This transforms text values into numerical columns
df_encoded = pd.get_dummies(df)


# Separate features (X) from the target variable (y)
X = df_encoded.drop(target_column, axis=1)
y = df_encoded[target_column]

# Store the feature column names for later use during prediction
feature_columns = X.columns

# FEATURE SCALING

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler to the dataset and transform the values
X_scaled = scaler.fit_transform(X)

# TRAIN TEST SPLIT

# Split the dataset into training and testing data
# 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,      # scaled feature dataset
    y,             # target variable
    test_size=0.2, # 20% data used for testing
    random_state=42  # random seed for reproducibility
)

# TRAIN MACHINE LEARNING MODELS

# Create a Linear Regression model
lr_model = LinearRegression()

# Train the Linear Regression model
lr_model.fit(X_train, y_train)


# Create a Decision Tree model
dt_model = DecisionTreeRegressor(

    # Maximum depth of the tree
    max_depth=10,

    # Random seed for consistent results
    random_state=42
)

# Train the Decision Tree model
dt_model.fit(X_train, y_train)


# Create a Random Forest model
rf_model = RandomForestRegressor(

    # Number of decision trees used
    n_estimators=100,

    # Maximum depth of each tree
    max_depth=10,

    # Random seed
    random_state=42
)

# Train the Random Forest model
rf_model.fit(X_train, y_train)

# MODEL EVALUATION

# Predict values using Linear Regression
lr_predictions = lr_model.predict(X_test)

# Calculate Linear Regression R² score
lr_score = r2_score(y_test, lr_predictions)


# Predict values using Decision Tree
dt_predictions = dt_model.predict(X_test)

# Calculate Decision Tree R² score
dt_score = r2_score(y_test, dt_predictions)


# Predict values using Random Forest
rf_predictions = rf_model.predict(X_test)

# Calculate Random Forest R² score
rf_score = r2_score(y_test, rf_predictions)

# SIDEBAR NAVIGATION

# Create a title in the sidebar
st.sidebar.title("Navigation")

# Create radio buttons to navigate between pages
page = st.sidebar.radio(

    "Select Section",

    [
        "Project Overview",
        "Dataset Exploration",
        "Machine Learning Models",
        "Prediction System"
    ]
)

# PAGE 1 : PROJECT OVERVIEW

if page == "Project Overview":

    # Display the main project title
    st.title("Student Performance Analytics System")

    # Explain the project purpose
    st.write("""
This project analyzes student academic performance using machine learning.

Different student factors such as study hours, attendance,
school type, and learning conditions are used to understand
how they affect student academic results.

Three machine learning models are trained and compared
to determine which model predicts student performance best.
""")

    # Create three columns to display statistics
    col1, col2, col3 = st.columns(3)

    # Show total number of students
    col1.metric("Total Students", 25000)

    # Show number of input features used for prediction
    col2.metric("Number of Columns", 16)

    # Show number of machine learning models used
    col3.metric("Models Compared", 3)

# PAGE 2 : DATASET EXPLORATION

elif page == "Dataset Exploration":

    # Display page title
    st.title("Dataset Exploration")

    # Show first few rows of the dataset
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Display statistical summary of numeric columns
    st.subheader("Statistical Summary")
    st.write(df.describe())

# PAGE 3 : MACHINE LEARNING MODELS

elif page == "Machine Learning Models":

    # Display title for model comparison
    st.title("Machine Learning Model Comparison")

    # Create three columns to show model scores
    col1, col2, col3 = st.columns(3)

    # Display Linear Regression performance
    col1.metric("Linear Regression R² Score", round(lr_score,3))

    # Display Decision Tree performance
    col2.metric("Decision Tree R² Score", round(dt_score,3))

    # Display Random Forest performance
    col3.metric("Random Forest R² Score", round(rf_score,3))

    # Determine the best model
    scores = {
        "Linear Regression": lr_score,
        "Decision Tree": dt_score,
        "Random Forest": rf_score
    }

    # Find the model with the highest score
    best_model = max(scores, key=scores.get)

    # Display best performing model
    st.success(f"The best performing model is: {best_model}")

# PAGE 4 : PREDICTION SYSTEM

elif page == "Prediction System":

    # Display prediction page title
    st.title("Predict Student Overall Score")

    # Provide instructions to the user
    st.write("""
Enter student information below.

The trained machine learning model will estimate
the student's expected overall academic score.
""")

    # Create dictionary to store user input
    input_data = {}

    # Loop through dataset columns to create input fields
    for col in df.columns:

        # Skip the target variable
        if col == target_column:
            continue

        # If the column contains text (categorical)
        if df[col].dtype == "object":

            # Create dropdown for category selection
            input_data[col] = st.selectbox(
                col,
                df[col].unique()
            )

        # If the column contains numbers
        else:

            # Create slider for numeric values
            input_data[col] = st.slider(
                col,
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean())
            )


    # When user presses prediction button
    if st.button("Predict Score"):

        # Convert input dictionary into dataframe
        input_df = pd.DataFrame([input_data])

        # Apply dummy variable encoding to match training data
        input_df = pd.get_dummies(input_df)

        # Align columns with training dataset
        input_df = input_df.reindex(columns=X.columns, fill_value=0)

        # Apply scaling using trained scaler
        input_scaled = scaler.transform(input_df)

        # Use Random Forest model to generate prediction
        prediction = rf_model.predict(input_scaled)[0]

        # Display predicted overall score
        st.success(f"Predicted Overall Score: {round(prediction,2)}")

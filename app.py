# Import Streamlit to build the web application interface
import streamlit as st

# Import pandas for handling and analyzing tabular data
import pandas as pd

# Import numpy for numerical operations
import numpy as np

# Import train_test_split to divide the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Import LabelEncoder to convert categorical text data into numbers
# Import StandardScaler to normalize numerical values
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import RandomForestRegressor which is an ensemble machine learning model
from sklearn.ensemble import RandomForestRegressor

# Import LinearRegression which is a simple baseline regression model
from sklearn.linear_model import LinearRegression

# Import r2_score to evaluate how well the models perform
from sklearn.metrics import r2_score


# Configure the Streamlit page layout and title
st.set_page_config(page_title="Student Performance Analytics", layout="wide")


# =====================================================
# LOAD DATA
# =====================================================

# Cache the data loading function so the dataset does not reload every time the app updates
@st.cache_data
def load_data():

    # Read the student dataset from the CSV file
    df = pd.read_csv("The_Real_Student_Performance.csv")

    # Remove hidden spaces from column names to avoid errors later
    df.columns = df.columns.str.strip()

    # Return the cleaned dataframe
    return df


# Load the dataset by calling the function
df = load_data()


# =====================================================
# REMOVE COLUMNS THAT SHOULD NOT BE USED
# =====================================================

# These columns are not useful for prediction
drop_columns = ["student_id", "final_grade"]

# Loop through the list of columns to remove them safely
for col in drop_columns:

    # Check if the column exists before removing it
    if col in df.columns:

        # Drop the column from the dataframe
        df = df.drop(col, axis=1)


# =====================================================
# DEFINE TARGET VARIABLE
# =====================================================

# The value we want the model to predict is the student's overall score
target_column = "overall_score"


# =====================================================
# DATA PREPROCESSING
# =====================================================

# Make a copy of the dataset for encoding operations
df_encoded = df.copy()

# Dictionary to store label encoders for each categorical column
label_encoders = {}


# Loop through columns that contain text values
for col in df_encoded.select_dtypes(include="object").columns:

    # Create a label encoder
    le = LabelEncoder()

    # Convert categorical values into numerical values
    df_encoded[col] = le.fit_transform(df_encoded[col])

    # Store the encoder so it can be reused later
    label_encoders[col] = le


# Separate input features from the target variable
X = df_encoded.drop(target_column, axis=1)

# Target variable the model will learn to predict
y = df_encoded[target_column]

# Store feature column names for later use in prediction
feature_columns = X.columns


# =====================================================
# FEATURE SCALING
# =====================================================

# Create a scaler to standardize numeric values
scaler = StandardScaler()

# Fit the scaler on the data and transform it
X_scaled = scaler.fit_transform(X)


# =====================================================
# SPLIT DATA INTO TRAIN AND TEST SETS
# =====================================================

# Divide the dataset into training and testing data
# 80% is used to train the model
# 20% is used to test model performance
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# =====================================================
# TRAIN MACHINE LEARNING MODELS
# =====================================================

# Create the Random Forest model
rf_model = RandomForestRegressor(

    # Number of trees used in the forest
    n_estimators=100,

    # Maximum depth of each tree
    max_depth=10,

    # Random seed for reproducibility
    random_state=42
)

# Train the Random Forest model using the training data
rf_model.fit(X_train, y_train)


# Create a Linear Regression model
lr_model = LinearRegression()

# Train the Linear Regression model
lr_model.fit(X_train, y_train)


# =====================================================
# MODEL EVALUATION
# =====================================================

# Evaluate Random Forest performance using R² score
rf_score = r2_score(y_test, rf_model.predict(X_test))

# Evaluate Linear Regression performance
lr_score = r2_score(y_test, lr_model.predict(X_test))


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

# Create a sidebar title for navigation
st.sidebar.title("Navigation")

# Create a radio button menu to switch between pages
page = st.sidebar.radio(
    "Select Section",
    [
        "Project Overview",
        "Dataset Exploration",
        "Machine Learning Models",
        "Prediction System"
    ]
)


# =====================================================
# PAGE 1 — PROJECT OVERVIEW
# =====================================================

if page == "Project Overview":

    # Display page title
    st.title("Student Performance Analytics System")

    # Explain the purpose of the project
    st.write("""
This project analyzes student academic performance using machine learning.

The system studies how different factors such as study hours,
attendance, school type, and student background influence
academic results.

Machine learning models are trained on the dataset in order
to predict the overall academic performance of students.
""")

    # Create three columns for displaying statistics
    col1, col2, col3 = st.columns(3)

    # Show total number of students
    col1.metric("Total Students", len(df))

    # Show number of features used in the model
    col2.metric("Number of Features", len(feature_columns))

    # Show number of machine learning models used
    col3.metric("Models Compared", 2)


# =====================================================
# PAGE 2 — DATASET EXPLORATION
# =====================================================

elif page == "Dataset Exploration":

    # Display page title
    st.title("Dataset Exploration")

    # Show first few rows of the dataset
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Display statistical summary of numerical columns
    st.subheader("Statistical Summary")
    st.write(df.describe())


# =====================================================
# PAGE 3 — MACHINE LEARNING MODELS
# =====================================================

elif page == "Machine Learning Models":

    # Page title
    st.title("Machine Learning Model Comparison")

    # Create two columns for displaying model scores
    col1, col2 = st.columns(2)

    # Display Random Forest performance score
    col1.metric("Random Forest R² Score", round(rf_score,3))

    # Display Linear Regression performance score
    col2.metric("Linear Regression R² Score", round(lr_score,3))

    # Display which model performed better
    if rf_score > lr_score:
        st.success("Random Forest performed better and was selected as the final model.")
    else:
        st.success("Linear Regression performed better.")


# =====================================================
# PAGE 4 — PREDICTION SYSTEM
# =====================================================

elif page == "Prediction System":

    # Page title
    st.title("Predict Student Overall Score")

    # Explanation text for the prediction system
    st.write("""
Enter student information below.
The model will estimate the student's expected overall academic score.
""")

    # Dictionary to store user inputs
    input_data = {}

    # Loop through each feature column
    for col in feature_columns:

        # If the column is categorical
        if col in label_encoders:

            # Get possible category options
            options = df[col].unique()

            # Create dropdown selection for the user
            input_data[col] = st.selectbox(
                f"{col}",
                options
            )

        # If the column is numerical
        else:

            # Determine slider range using dataset values
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())

            # Create slider for numeric input
            input_data[col] = st.slider(
                f"{col}",
                min_val,
                max_val,
                mean_val
            )


    # When the user clicks the prediction button
    if st.button("Predict Score"):

        # Convert input dictionary into a dataframe
        input_df = pd.DataFrame([input_data])

        # Apply label encoding to categorical inputs
        for col in label_encoders:
            if col in input_df:
                input_df[col] = label_encoders[col].transform(input_df[col])

        # Ensure column order matches the training data
        input_df = input_df[feature_columns]

        # Scale the input values using the trained scaler
        input_scaled = scaler.transform(input_df)

        # Use the Random Forest model to predict the score
        prediction = rf_model.predict(input_scaled)[0]

        # Display the predicted score
        st.success(f"Predicted Overall Score: {round(prediction,2)}")

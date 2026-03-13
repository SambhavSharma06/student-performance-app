
# IMPORT LIBRARIES

import streamlit as st        # Import Streamlit to build the web application interface
import pandas as pd           # Import pandas to handle and manipulate tabular data
import numpy as np            # Import numpy for numerical operations

from sklearn.model_selection import train_test_split  # Import function to split dataset into training and testing
from sklearn.preprocessing import StandardScaler      # Import scaler to normalize numeric values

from sklearn.linear_model import LogisticRegression   # Import Logistic Regression classification model
from sklearn.tree import DecisionTreeClassifier       # Import Decision Tree classification model
from sklearn.ensemble import RandomForestClassifier   # Import Random Forest classification model

# PAGE SETTINGS

st.set_page_config(                               # Configure settings for the Streamlit web page
    page_title="Student Performance Analytics",   # Title shown in the browser tab
    page_icon="🎓",                                # Icon shown in the browser tab
    layout="wide"                                  # Use wide layout to give more space on the page
)

# LOAD DATA

@st.cache_data                                    # Cache the dataset so it loads faster when the app reloads
def load_data():                                   # Define a function to load the dataset

    df = pd.read_csv("The_Real_Student_Performance.csv")   # Read the CSV dataset file

    df.columns = df.columns.str.strip()             # Remove any hidden spaces from column names

    return df                                       # Return the cleaned dataframe


df = load_data()                                    # Call the function to load the dataset

# REMOVE USELESS COLUMNS

if "student_id" in df.columns:                      # Check if student_id column exists
    df = df.drop("student_id", axis=1)              # Remove student_id because it does not help prediction

# remove overall score because it leaks information
if "overall_score" in df.columns:                   # Check if overall_score column exists
    df = df.drop("overall_score", axis=1)           # Remove it because it already summarizes subject scores

# TARGET VARIABLE

target_column = "final_grade"                       # Define the target variable the model will predict

# SPLIT INPUT AND OUTPUT

X = df.drop(target_column, axis=1)                  # X contains all input variables except the target column

y = df[target_column]                               # y contains the target variable (final_grade)

# DUMMY VARIABLES

X = pd.get_dummies(X)                               # Convert categorical columns into numerical dummy variables

feature_columns = X.columns                         # Store the list of feature columns after encoding

# SCALE DATA

scaler = StandardScaler()                           # Create a StandardScaler object

X_scaled = scaler.fit_transform(X)                  # Scale the dataset so all variables have similar ranges

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(  # Split the dataset into training and testing sets
    X_scaled,                                          # Use the scaled features as input
    y,                                                 # Target variable
    test_size=0.2,                                     # Use 20% of data for testing
    random_state=42                                    # Set random seed for reproducibility
)

# TRAIN MODELS

lr_model = LogisticRegression(max_iter=1000)         # Create Logistic Regression model with increased iterations
lr_model.fit(X_train, y_train)                       # Train the Logistic Regression model

dt_model = DecisionTreeClassifier(random_state=42)   # Create Decision Tree classifier
dt_model.fit(X_train, y_train)                       # Train the Decision Tree model

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Create Random Forest with 100 trees
rf_model.fit(X_train, y_train)                       # Train the Random Forest model

# SIDEBAR

st.sidebar.title("📊 Navigation")                     # Create sidebar title for navigation menu

page = st.sidebar.radio(                              # Create radio buttons in sidebar
    "Select Section",                                  # Label of the sidebar menu
    [
        "Project Overview",                            # Page 1
        "Dataset Exploration",                         # Page 2 
        "Machine Learning Models",                     # Page 3 
        "Prediction System"                            # Page 4
    ]
)

# PAGE 1

if page == "Project Overview":                         # Check if the user selected the overview page

    st.title("🎓 Student Performance Analytics System") # Display main project title

    st.write("""                                       # Display project description text
This project analyzes student academic performance using machine learning.

The system studies how factors such as study hours, attendance,
school type, internet access, and learning methods influence
student academic results.

Three machine learning models were trained and compared
to determine which model predicts student final grades
with the highest accuracy.
""")

    col1, col2, col3 = st.columns(3)                   # Create three columns for metrics display

    col1.metric("Total Students", len(df))             # Show number of students in the dataset
    col2.metric("Total Columns", 16)                   # Show number of dataset columns
    col3.metric("Models Compared", 3)                  # Show number of machine learning models used

# PAGE 2

elif page == "Dataset Exploration":                    # Check if dataset exploration page is selected

    st.title("📊 Dataset Exploration")                 # Display page title

    st.subheader("Dataset Preview")                    # Show preview section title

    st.dataframe(df.head())                            # Display the first few rows of the dataset

    st.subheader("Statistical Summary")                # Section title for statistical analysis

    st.write(df.describe())                            # Display summary statistics of numeric columns

    st.subheader("Distribution of Final Grades")       # Section title for grade distribution

    grade_counts = df["final_grade"].value_counts()    # Count how many students fall into each grade

    st.bar_chart(grade_counts)                         # Display bar chart showing grade distribution

# PAGE 3

elif page == "Machine Learning Models":                # Check if model comparison page is selected

    st.title("🤖 Machine Learning Model Comparison")   # Display page title

    st.subheader("Accuracy Score (All Models)")        # Section title for model performance

    st.code("""                                        # Display model accuracy results
Logistic Regression Accuracy: 0.7632
Decision Tree Accuracy: 0.8600
Random Forest Accuracy: 0.9024
""")

    st.success("Random Forest achieved the highest accuracy and was selected as the final model.")  # Highlight best model

    model_data = pd.DataFrame({                        # Create dataframe containing model accuracy values
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Accuracy": [0.7632, 0.86, 0.9024]
    })

    st.subheader("Model Accuracy Comparison")          # Title for model comparison chart

    st.bar_chart(model_data.set_index("Model"))        # Display bar chart comparing model accuracy

# PAGE 4

elif page == "Prediction System":                      # Check if prediction page is selected

    st.title("🎯 Predict Student Final Grade")         # Display prediction page title

    st.write("""                                       # Display instructions for the user
Enter student information below.

The trained Random Forest model will estimate
the student's expected final grade.
""")

    input_data = {}                                    # Create empty dictionary to store user inputs

    for col in df.columns:                             # Loop through all dataset columns

        if col == target_column:                       # Skip the target column
            continue

        if df[col].dtype == "object":                  # If the column is categorical

            input_data[col] = st.selectbox(            # Create dropdown selection for categorical input
                col.replace("_"," ").title(),          # Format column name nicely for UI
                df[col].unique()                       # Show all unique values as options
            )

        else:                                          # If the column is numeric

            input_data[col] = st.slider(               # Create slider for numeric input
                col.replace("_"," ").title(),          # Format column name nicely
                float(df[col].min()),                  # Minimum slider value
                float(df[col].max()),                  # Maximum slider value
                float(df[col].mean())                  # Default slider value
            )


    if st.button("Predict Grade"):                     # When user clicks the prediction button

        input_df = pd.DataFrame([input_data])          # Convert user inputs into dataframe

        input_df = pd.get_dummies(input_df)            # Apply dummy variable encoding

        input_df = input_df.reindex(                   # Align columns with training dataset
            columns=feature_columns,
            fill_value=0
        )

        input_scaled = scaler.transform(input_df)      # Apply same scaling used during training

        prediction = rf_model.predict(input_scaled)[0] # Predict the final grade using Random Forest model

        st.success(f"Predicted Final Grade: {prediction}") # Display the predicted grade

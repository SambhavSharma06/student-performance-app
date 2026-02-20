import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Student Performance Analytics", layout="wide")

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_csv("The_Real_Student_Performance.csv")
    df.columns = df.columns.str.strip()  # remove hidden spaces
    return df

df = load_data()

# Automatically detect math score column
target_column = [col for col in df.columns if "math" in col.lower()][0]

# =====================================================
# ENCODE DATA
# =====================================================

df_encoded = df.copy()

label_encoders = {}
for col in df_encoded.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

X = df_encoded.drop(target_column, axis=1)
y = df_encoded[target_column]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =====================================================
# TRAIN MODELS
# =====================================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_score = r2_score(y_test, rf_model.predict(X_test))
lr_score = r2_score(y_test, lr_model.predict(X_test))

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Project Overview", "Data Exploration", "Machine Learning Models"]
)

# =====================================================
# PAGE 1 — PROJECT OVERVIEW
# =====================================================

if page == "Project Overview":

    st.title("🎓 Student Performance Analytics System")

    st.markdown("""
    This project analyzes student academic performance using machine learning.

    The system explores how demographic, social, and academic factors influence 
    student math scores and compares different regression models to determine 
    which provides the most accurate predictions.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Total Features", X.shape[1])
    col3.metric("Models Compared", 2)

    st.subheader("Methodology")

    st.markdown("""
    - Data Cleaning & Preprocessing  
    - Encoding Categorical Variables  
    - Feature Scaling using StandardScaler  
    - Training Regression Models  
    - Evaluating Performance using R² Score  
    """)

# =====================================================
# PAGE 2 — DATA EXPLORATION
# =====================================================

elif page == "Data Exploration":

    st.title("📊 Data Exploration & Filtering")

    st.markdown("""
    Use the filters below to explore how different groups of students perform.
    """)

    filtered_df = df.copy()

    # Dynamic filters for categorical columns
    categorical_columns = df.select_dtypes(include="object").columns

    for col in categorical_columns:
        options = ["All"] + list(df[col].unique())
        selection = st.selectbox(f"Filter by {col}", options)
        if selection != "All":
            filtered_df = filtered_df[filtered_df[col] == selection]

    st.subheader("Filtered Dataset")
    st.dataframe(filtered_df)

    st.subheader("Average Scores")
    numeric_cols = filtered_df.select_dtypes(include=np.number).columns
    st.write(filtered_df[numeric_cols].mean())

# =====================================================
# PAGE 3 — MACHINE LEARNING MODELS
# =====================================================

elif page == "Machine Learning Models":

    st.title("🤖 Machine Learning Model Comparison")

    col1, col2 = st.columns(2)

    col1.metric("Random Forest R² Score", round(rf_score, 3))
    col2.metric("Linear Regression R² Score", round(lr_score, 3))

    if rf_score > lr_score:
        best_model = "Random Forest"
    else:
        best_model = "Linear Regression"

    st.success(f"Best Performing Model: {best_model}")

    st.subheader("Predict Student Math Score")

    input_data = []

    for col in X.columns:
        value = st.number_input(f"Enter {col}", value=0.0)
        input_data.append(value)

    if st.button("Predict Final Score"):

        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        prediction = rf_model.predict(input_scaled)[0]

        st.success(f"Predicted Math Score: {round(prediction, 2)}")

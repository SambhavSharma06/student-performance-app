import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Student Performance Analytics", layout="wide")

# ==============================
# Load Dataset
# ==============================

@st.cache_data
def load_data():
    df = pd.read_csv("The_Real_Student_Performance.csv")
    return df

df = load_data()

# ==============================
# Encode Data
# ==============================

df_encoded = df.copy()

label_encoders = {}
for col in df_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

X = df_encoded.drop("math score", axis=1)
y = df_encoded["math score"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==============================
# Train Models
# ==============================

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

# ==============================
# Sidebar Navigation
# ==============================

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Project Overview", "Data Exploration", "Machine Learning Models"]
)

# ==============================
# PAGE 1 — OVERVIEW
# ==============================

if page == "Project Overview":

    st.title("🎓 Student Performance Analytics System")

    st.markdown("""
    This project analyzes student academic performance and uses machine learning
    models to predict final math scores based on various demographic and academic factors.

    The objective is to understand how different variables influence student performance
    and determine which machine learning model provides the best predictive accuracy.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Features Used", X.shape[1])
    col3.metric("Models Trained", 2)

    st.subheader("Project Approach")

    st.markdown("""
    1. Data preprocessing and encoding categorical variables  
    2. Feature scaling using StandardScaler  
    3. Training multiple regression models  
    4. Comparing performance using R² Score  
    5. Selecting best-performing model  
    """)

# ==============================
# PAGE 2 — DATA EXPLORATION
# ==============================

elif page == "Data Exploration":

    st.title("📊 Data Exploration & Filters")

    st.markdown("""
    Use the filters below to explore how different student groups perform.
    """)

    gender_filter = st.selectbox(
        "Filter by Gender",
        ["All"] + list(df["gender"].unique())
    )

    race_filter = st.selectbox(
        "Filter by Race/Ethnicity",
        ["All"] + list(df["race/ethnicity"].unique())
    )

    parental_filter = st.selectbox(
        "Filter by Parental Education",
        ["All"] + list(df["parental level of education"].unique())
    )

    filtered_df = df.copy()

    if gender_filter != "All":
        filtered_df = filtered_df[filtered_df["gender"] == gender_filter]

    if race_filter != "All":
        filtered_df = filtered_df[filtered_df["race/ethnicity"] == race_filter]

    if parental_filter != "All":
        filtered_df = filtered_df[filtered_df["parental level of education"] == parental_filter]

    st.write("Filtered Data")
    st.dataframe(filtered_df)

    st.subheader("Average Scores")

    st.write(filtered_df[["math score", "reading score", "writing score"]].mean())

# ==============================
# PAGE 3 — MACHINE LEARNING
# ==============================

elif page == "Machine Learning Models":

    st.title("🤖 Machine Learning Models")

    st.subheader("Model Comparison")

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

    if st.button("Predict Final Grade"):
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        prediction = rf_model.predict(input_scaled)[0]

        st.success(f"Predicted Math Score: {round(prediction, 2)}")

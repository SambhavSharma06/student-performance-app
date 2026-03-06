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
    df.columns = df.columns.str.strip()
    return df

df = load_data()

target_column = "math_score"

# =====================================================
# PREPROCESS DATA
# =====================================================

df_encoded = df.copy()

label_encoders = {}

for col in df_encoded.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

X = df_encoded.drop(target_column, axis=1)
y = df_encoded[target_column]

feature_columns = X.columns

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
# SIDEBAR
# =====================================================

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

# =====================================================
# PAGE 1
# =====================================================

if page == "Project Overview":

    st.title("🎓 Student Performance Analytics System")

    st.write("""
This project analyzes student academic performance using machine learning.

The goal of this system is to understand how different factors such as
study hours, attendance, school type, and student background influence
academic results.

The project trains machine learning models and compares them to find
which model predicts student math scores most accurately.
""")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Number of Features", X.shape[1])
    col3.metric("Models Used", 2)

# =====================================================
# PAGE 2
# =====================================================

elif page == "Dataset Exploration":

    st.title("📊 Dataset Exploration")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Statistical Summary")
    st.write(df.describe())

# =====================================================
# PAGE 3
# =====================================================

elif page == "Machine Learning Models":

    st.title("🤖 Machine Learning Model Evaluation")

    col1, col2 = st.columns(2)

    col1.metric("Random Forest R² Score", round(rf_score,3))
    col2.metric("Linear Regression R² Score", round(lr_score,3))

    if rf_score > lr_score:
        st.success("Random Forest performed better and was selected as the final model.")
    else:
        st.success("Linear Regression performed better.")

# =====================================================
# PAGE 4
# =====================================================

elif page == "Prediction System":

    st.title("🔮 Predict Student Math Score")

    st.write("""
Enter student information below. The system will use the trained
machine learning model to estimate the expected math score.
""")

    input_data = {}

    for col in feature_columns:

        if col in label_encoders:

            options = df[col].unique()

            input_data[col] = st.selectbox(
                f"{col}",
                options
            )

        else:

            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())

            input_data[col] = st.slider(
                f"{col}",
                min_val,
                max_val,
                mean_val
            )

    if st.button("Predict Score"):

        input_df = pd.DataFrame([input_data])

        # Encode categorical values
        for col in label_encoders:
            if col in input_df:
                input_df[col] = label_encoders[col].transform(input_df[col])

        # Ensure correct column order
        input_df = input_df[feature_columns]

        input_scaled = scaler.transform(input_df)

        prediction = rf_model.predict(input_scaled)[0]

        st.success(f"Predicted Math Score: {round(prediction,2)}")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Student Performance Dashboard",
    layout="wide"
)

st.title("📊 Student Performance Analytics Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("The_Real_Student_Performance.csv")

df = load_data()

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("🔎 Filters")

selected_gender = st.sidebar.multiselect(
    "Select Gender",
    options=df["gender"].unique(),
    default=df["gender"].unique()
)

selected_school = st.sidebar.multiselect(
    "Select School Type",
    options=df["school_type"].unique(),
    default=df["school_type"].unique()
)

selected_internet = st.sidebar.multiselect(
    "Internet Access",
    options=df["internet_access"].unique(),
    default=df["internet_access"].unique()
)

study_range = st.sidebar.slider(
    "Study Hours",
    float(df["study_hours"].min()),
    float(df["study_hours"].max()),
    (float(df["study_hours"].min()), float(df["study_hours"].max()))
)

attendance_range = st.sidebar.slider(
    "Attendance Percentage",
    int(df["attendance_percentage"].min()),
    int(df["attendance_percentage"].max()),
    (int(df["attendance_percentage"].min()), int(df["attendance_percentage"].max()))
)

selected_grade = st.sidebar.multiselect(
    "Final Grade",
    options=df["final_grade"].unique(),
    default=df["final_grade"].unique()
)

# -----------------------------
# APPLY FILTERS
# -----------------------------
filtered_df = df[
    (df["gender"].isin(selected_gender)) &
    (df["school_type"].isin(selected_school)) &
    (df["internet_access"].isin(selected_internet)) &
    (df["study_hours"].between(study_range[0], study_range[1])) &
    (df["attendance_percentage"].between(attendance_range[0], attendance_range[1])) &
    (df["final_grade"].isin(selected_grade))
]

# -----------------------------
# SUMMARY METRICS
# -----------------------------
st.subheader("📈 Summary Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Students", len(filtered_df))
col2.metric("Average Overall Score", round(filtered_df["overall_score"].mean(), 2))
col3.metric("Average Study Hours", round(filtered_df["study_hours"].mean(), 2))

# -----------------------------
# VISUALIZATIONS
# -----------------------------
st.subheader("📊 Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    sns.histplot(filtered_df["overall_score"], bins=20, kde=True, ax=ax1)
    ax1.set_title("Distribution of Overall Score")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.boxplot(x="final_grade", y="attendance_percentage", data=filtered_df, ax=ax2)
    ax2.set_title("Attendance by Final Grade")
    st.pyplot(fig2)

# Correlation Heatmap
st.subheader("🔥 Correlation Heatmap")

numeric_cols = filtered_df.select_dtypes(include=np.number)

fig3, ax3 = plt.subplots(figsize=(8,6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)

# -----------------------------
# MODEL PREDICTION SECTION
# -----------------------------
st.subheader("🤖 Predict Student Final Grade")

try:
    rf_model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 10, 25, 16)
        study_hours = st.number_input("Study Hours", 0.0, 15.0, 5.0)
        attendance = st.number_input("Attendance Percentage", 0, 100, 80)
        math_score = st.number_input("Math Score", 0, 100, 70)
        science_score = st.number_input("Science Score", 0, 100, 70)
        english_score = st.number_input("English Score", 0, 100, 70)

    with col2:
        gender = st.selectbox("Gender", df["gender"].unique())
        school_type = st.selectbox("School Type", df["school_type"].unique())
        internet_access = st.selectbox("Internet Access", df["internet_access"].unique())

    if st.button("Predict Grade"):

        input_data = pd.DataFrame([{
            "age": age,
            "study_hours": study_hours,
            "attendance_percentage": attendance,
            "math_score": math_score,
            "science_score": science_score,
            "english_score": english_score,
            "gender": gender,
            "school_type": school_type,
            "internet_access": internet_access
        }])

        # Encode categorical same way as training
        input_data = pd.get_dummies(input_data)

        # Align columns with training data
        model_columns = rf_model.feature_names_in_
        input_data = input_data.reindex(columns=model_columns, fill_value=0)

        # Scale
        input_scaled = scaler.transform(input_data)

        prediction = rf_model.predict(input_scaled)

        st.success(f"Predicted Final Grade: {prediction[0]}")

except:
    st.warning("Model files not found. Please upload rf_model.pkl and scaler.pkl.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("Developed for Data Analytics College Assignment")

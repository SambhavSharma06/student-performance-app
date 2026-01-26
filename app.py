import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Page config ----------
st.set_page_config(
    page_title="Student Performance Dashboard",
    layout="wide",
)

# ---------- Load data ----------
df = pd.read_csv("The_Real_Student_Performance.csv")

# ---------- Sidebar ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Data Analysis", "Model Information"]
)

# ---------- Header ----------
st.title("🎓 Student Performance Analysis Dashboard")
st.write(
    "An interactive data analytics dashboard to explore student performance "
    "and academic patterns using real-world educational data."
)

# ---------- DASHBOARD ----------
if page == "Dashboard":
    st.subheader("📊 Key Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Students", df.shape[0])

    with col2:
        st.metric("Average Overall Score", round(df["overall_score"].mean(), 2))

    with col3:
        pass_rate = (df["final_grade"].isin(["a", "b", "c"])).mean() * 100
        st.metric("Pass Rate (%)", round(pass_rate, 2))

    st.markdown("---")
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ---------- DATA ANALYSIS ----------
elif page == "Data Analysis":
    st.subheader("🔍 Exploratory Data Analysis")

    # Filters
    st.markdown("### Filters")
    col1, col2 = st.columns(2)

    with col1:
        gender_filter = st.multiselect(
            "Select Gender",
            df["gender"].unique(),
            default=df["gender"].unique()
        )

    with col2:
        school_filter = st.multiselect(
            "Select School Type",
            df["school_type"].unique(),
            default=df["school_type"].unique()
        )

    filtered_df = df[
        (df["gender"].isin(gender_filter)) &
        (df["school_type"].isin(school_filter))
    ]

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Final Grade Distribution")
        fig1, ax1 = plt.subplots()
        sns.countplot(x="final_grade", data=filtered_df, ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.markdown("### Study Hours vs Overall Score")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(
            x="study_hours",
            y="overall_score",
            data=filtered_df,
            alpha=0.4,
            ax=ax2
        )
        st.pyplot(fig2)

# ---------- MODEL INFO ----------
else:
    st.subheader("🤖 Machine Learning Model")

    st.markdown(
        """
        **Problem Type:** Supervised Classification  
        **Target Variable:** Final Grade  
        """
    )

    st.markdown("### Models Trained")
    st.write(
        "- Logistic Regression\n"
        "- Decision Tree\n"
        "- Random Forest"
    )

    st.markdown("### Final Model Selected")
    st.success(
        "Random Forest Classifier was selected due to highest accuracy "
        "and balanced precision-recall performance."
    )

    st.markdown("### Why Random Forest?")
    st.write(
        "Random Forest performs well on complex datasets by combining multiple "
        "decision trees, reducing overfitting and improving generalization."
    )

# ---------- Footer ----------
st.markdown("---")
st.caption("Developed as part of a Data Analytics & Machine Learning Project")

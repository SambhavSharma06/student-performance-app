# student-performance-app

## Project Overview

This project aims to predict student final grades using machine learning techniques.  
The objective is to analyse various academic and behavioural factors such as study hours, attendance, and subject scores, and determine their impact on final academic performance.

The project includes:
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature scaling
- Model training and evaluation
- Web application deployment using Streamlit

## Dataset

The dataset used in this project is:

The_Real_Student_Performance.csv

It contains information about:
- Study hours
- Attendance percentage
- Subject scores (Math, Science, English)
- Internet access
- Final grade (Target variable)

The target variable is **final_grade**, which is treated as a classification problem.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Machine Learning Models Used

The following models were trained and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

The models were evaluated using:
- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix
- 
## Data Preprocessing Steps

1. Removed unnecessary columns.
2. Encoded categorical variables using dummy encoding.
3. Split dataset into training (80%) and testing (20%) sets.
4. Applied StandardScaler AFTER splitting to prevent data leakage.
5. Trained models using training data only.

## Model Evaluation

The models were compared based on accuracy and classification reports.

Among the tested models, Random Forest achieved the highest accuracy and showed better performance in handling non-linear relationships.

Therefore, Random Forest was selected as the final model for deployment.


## Web Application (Streamlit)

A web application was built using Streamlit to allow users to input student details and predict the final grade.

The trained model and scaler were saved using:

- rf_model.pkl
- scaler.pkl

These files are loaded inside the Streamlit app to generate predictions.


## How to Run the Project Locally

1. Clone the repository
2. Install dependencies


3. Run the Streamlit app

This project demonstrates how machine learning can be applied to educational data to predict student performance.  

It highlights the importance of:
- Study hours
- Attendance
- Subject consistency

The project combines data analysis, model development, and deployment into a complete end-to-end system. 

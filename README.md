# customer_churn_prediction

This project focuses on predicting **customer churn** using machine learning techniques and deploying the model as an interactive **Streamlit web application**. Customer churn prediction helps businesses identify customers who are likely to leave and take proactive retention measures.
## Problem Statement
Customer churn is a major challenge for service-based industries. The goal of this project is to analyze customer behavior and predict whether a customer will **churn (Yes)** or **stay (No)** based on their service usage and account information.
## Dataset
- **Dataset Name:** Telco Customer Churn Dataset  
- **Source:** IBM Sample Datasets  
- **Target Variable:** `Churn` (Yes / No)
### Key Features:
- Customer tenure
- Contract type
- Monthly and total charges
- Internet & phone services
- Payment method
- Support services
## Project Workflow
1. Data loading and preprocessing  
2. Handling missing values and categorical encoding  
3. Exploratory Data Analysis (EDA)  
4. Feature selection  
5. Machine Learning model training  
6. Model evaluation  
7. Deployment using Streamlit  
## Machine Learning Model
- Algorithm: Logistic Regression / Random Forest (based on implementation)
- Evaluation Metrics:
  - Accuracy
  - Confusion Matrix
  - Classification Report
##  Streamlit Web App
The Streamlit app allows users to:
- Input customer details
- Predict whether the customer will churn or stay
- Interactively test different customer scenarios

### Run the app locally:
streamlit run app.py


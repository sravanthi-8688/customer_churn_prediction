import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ------------------------------
# Load and preprocess dataset
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\sravanthi\\OneDrive\\Pictures\\Desktop\\WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df.dropna(inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    return df

df = load_data()

# Split data
X = df.drop("Churn", axis=1)
y = df["Churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Customer Churn Prediction Dashboard")
st.markdown("This app predicts customer churn and also shows churn insights & feature importance.")

# Sidebar input fields
st.sidebar.header("Enter Customer Information")

def user_input():
    customer = {}
    for col in df.drop("Churn", axis=1).columns:
        val = st.sidebar.number_input(f"{col}", value=float(df[col].median()))
        customer[col] = val
    return pd.DataFrame([customer])

input_df = user_input()

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction Result")
st.write("🔴 **Customer will Churn**" if prediction[0] == 1 else "🟢 **Customer will Stay**")
st.write("Probability of Churn:", round(prediction_proba[0][1] * 100, 2), "%")

# ------------------------------
# Analytics Section
# ------------------------------
st.subheader(" Churn Data Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Churn Distribution**")
    fig, ax = plt.subplots()
    sns.countplot(x=y, palette="coolwarm", ax=ax)
    ax.set_xticklabels(["Stay", "Churn"])
    st.pyplot(fig)

with col2:
    st.markdown("**Churn Rate (%)**")
    churn_rate = (y.sum() / len(y)) * 100
    st.metric(label="Overall Churn Rate", value=f"{churn_rate:.2f}%")

# Feature Importance
st.subheader("Top 10 Features Influencing Churn")
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x=importances[indices][:10], y=features[indices][:10], palette="viridis", ax=ax)
st.pyplot(fig)

# Correlation Heatmap
st.subheader("Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False, ax=ax)
st.pyplot(fig)

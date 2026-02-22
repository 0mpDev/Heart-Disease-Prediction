import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load best model and scaler
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below to check the likelihood of heart disease using the best trained model.")

# ======================
# Input fields
# ======================
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", 20, 100)
    sex = st.selectbox("Sex", ["Male", "Female"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200)
with col2:
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
    cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
with col3:
    restecg = st.selectbox("Resting ECG Results", ["normal", "st-t abnormality", "left ventricular hypertrophy"])
    thalch = st.number_input("Max Heart Rate Achieved", 60, 220)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])

oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, step=0.1)

# ======================
# Convert to numeric
# ======================
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# Create DataFrame
user_df = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalch': [thalch],
    'exang': [exang],
    'oldpeak': [oldpeak]
})

# ======================
# Apply one-hot encoding (same as training)
# ======================
user_df = pd.get_dummies(user_df, columns=['cp', 'restecg'], drop_first=True)

# Ensure all columns match training set
expected_cols = model.feature_names_in_
for col in expected_cols:
    if col not in user_df.columns:
        user_df[col] = 0
user_df = user_df[expected_cols]

# ======================
# Scale numeric features
# ======================
numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
user_df[numeric_cols] = scaler.transform(user_df[numeric_cols])

# ======================
# Prediction Button
# ======================
if st.button("🔍 Predict"):
    pred = model.predict(user_df)[0]
    proba = model.predict_proba(user_df)[0][1] if hasattr(model, "predict_proba") else None

    st.subheader("🩺 Prediction Result")
    if pred == 1:
        st.error("⚠️ The patient is **likely to have heart disease.**")
    else:
        st.success("✅ The patient is **unlikely to have heart disease.**")

    if proba is not None:
        st.progress(float(proba))
        st.write(f"**Prediction Confidence:** {proba*100:.2f}%")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.title("Heart Disease Prediction App ❤️")

# Load dataset
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

df = pd.read_csv("processed.cleveland.data", names=columns)

# Data cleaning
df.replace("?", pd.NA, inplace=True)
df = df.dropna()
df = df.astype(float)

# Target fix
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# Features & labels
X = df.drop("target", axis=1)
y = df["target"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Training Model
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# Inputs UI
st.sidebar.header("Enter Patient Details")

age = st.sidebar.number_input("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.sidebar.number_input("Resting BP", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 (1=True, 0=False)", [1,0])
restecg = st.sidebar.selectbox("Rest ECG (0-2)", [0,1,2])
thalach = st.sidebar.number_input("Max Heart Rate", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", [1,0])
oldpeak = st.sidebar.number_input("Oldpeak", 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox("Slope (0-2)", [0,1,2])
ca = st.sidebar.selectbox("Major Vessels (0-3)", [0,1,2,3])
thal = st.sidebar.selectbox("Thal (0-3)", [0,1,2,3])

# Prediction button
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"High Risk ⚠️\nProbability: {probability:.2f}")
        st.write("Please consult a doctor for further diagnosis.")
    else:
        st.success(f"Low Risk ✅\nProbability: {probability:.2f}")
        st.write("Maintain a healthy lifestyle!")
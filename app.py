# app.py
import streamlit as st
import numpy as np
import pickle

# Load trained model
with open("models/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient health details to predict diabetes risk.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=30)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=100)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Predict button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                             skin_thickness, insulin, bmi, dpf, age]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High risk of Diabetes")
    else:
        st.success("✅ Low risk of Diabetes")

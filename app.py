import streamlit as st
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load("depression_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🧠 Depression Prediction App")
st.write("Fill in the details to check if the person might be experiencing depression.")

# Input features from user
age = st.slider("Age", 12, 60, 20)
work_study_hours = st.slider("Work/Study Hours", 0, 18, 6)

financial_stress = st.selectbox("Do you have Financial Stress?", ["No", "Yes"])
financial_stress = 1 if financial_stress == "Yes" else 0

sleep_duration = st.selectbox("Sleep Duration", ["<5", "5-6", "6-7", "7-8", "8+"])
job_satisfaction = st.slider("Job Satisfaction (1 to 10)", 1, 10, 5)
work_pressure = st.slider("Work Pressure (1 to 10)", 1, 10, 5)

# Convert categorical to numerical (if needed)
sleep_dict = {"<5": 0, "5-6": 1, "6-7": 2, "7-8": 3, "8+": 4}
sleep_duration = sleep_dict[sleep_duration]

# Collect inputs in correct order
input_data = np.array([[age, work_pressure, job_satisfaction, sleep_duration, 
                        work_study_hours, financial_stress]])

# Scale the input
scaled_input = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(scaled_input)
    result = "Likely Depressed 😟" if prediction[0] == 1 else "Not Depressed 🙂"
    st.subheader(f"Prediction: {result}")

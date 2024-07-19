#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained model
model = joblib.load('model_1.pkl')

scaler = StandardScaler()

# Streamlit app layout
def main():
    st.title("Maternal Health Risk Prediction")

    # Description
    st.write("""
    This application predicts the risk level based on patient details. 
    Please enter the details below and click "Predict Risk Level" to see the results.
    """)

    def predict_risk_level(age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate):

        # Create a DataFrame from the input values
        input_data = pd.DataFrame({
        'Age': [age],
        'SystolicBP': [systolic_bp],
        'DiastolicBP': [diastolic_bp],
        'BS': [bs],
        'BodyTemp': [body_temp],
        'HeartRate': [heart_rate]
    })

        # The same scaling used during training
        input_data_scaled = scaler.fit_transform(input_data)

        # Predict
        prediction = model.predict(input_data_scaled)

        return prediction





    # Create a form for user input
    with st.form(key='patient_form'):
        st.subheader("Patient Information")

        # Layout with columns
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            systolic_bp = st.number_input("Systolic Blood Pressure", min_value=50, max_value=200, value=120)
            diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=30, max_value=120, value=80)
            bs = st.number_input("Blood Sugar (mg/dL)", min_value=0.0, max_value=20.0, step=0.1, value=5.5)

        with col2:
            body_temp = st.number_input("Body Temperature (°F)", min_value=32.0, max_value=113.0, step=0.1, value=98.6)
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=70)

        # Submit button
        submit_button = st.form_submit_button(label='Predict Risk Level')

    # Prepare patient info for prediction when the submit button is clicked
    if submit_button:
        st.spinner()

        prediction = predict_risk_level(age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate)

        # Display prediction
        st.subheader("Prediction Result:")
        if prediction[0] == 0:
            st.warning("The patient is predicted to be at low risk.")
        elif prediction[0] == 1:
            st.success("The patient is predicted to be at mid risk.")
        else:
            st.error("The patient is predicted to be at high risk.")

if __name__ == "__main__":
    main()


# In[ ]:





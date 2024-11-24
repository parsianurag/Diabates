import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the scaler and model
scaler = pickle.load(open("Model/standardScalar.pkl", "rb"))
model = pickle.load(open("Model/modelForPrediction.pkl", "rb"))

# Streamlit app title
st.title("Diabetes Prediction")

# User input form
st.header("Enter Patient Details")

# Input fields for user data
Pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
Glucose = st.number_input("Glucose", min_value=0.0, step=0.1)
BloodPressure = st.number_input("Blood Pressure", min_value=0.0, step=0.1)
SkinThickness = st.number_input("Skin Thickness", min_value=0.0, step=0.1)
Insulin = st.number_input("Insulin", min_value=0.0, step=0.1)
BMI = st.number_input("BMI", min_value=0.0, step=0.1)
DiabetesPredigeeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
Age = st.number_input("Age", min_value=0, step=1)

# Predict button
if st.button("Predict"):
    # Prepare the input data for scaling and prediction
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPredigeeFunction, Age]])
    new_data = scaler.transform(input_data)
    
    # Make prediction
    predict = model.predict(new_data)
    
    # Display result
    if predict[0] == 1:
        st.success("Prediction: Diabetic")
    else:
        st.success("Prediction: Non-Diabetic")

with st.container():
    right_column, left_column = st.columns(2)
    with left_column:

        st.write('_For any issue contact me via:_')
        st.info('[LinkedIn](https://www.linkedin.com/in/anurag-parsi-407377238)', icon="📩")
        st.info('[anuragdscon@gmail.com]', icon="📩")

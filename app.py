# app.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
import joblib  # Import joblib for loading the model

# Load the trained model and scaler
ann_model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit application
st.title("Breast Cancer Prediction")

# User input for patient characteristics
st.header("Enter Patient Characteristics")

# Load the breast cancer dataset for feature names
data = load_breast_cancer()

# Create input fields for each feature
features = {}
for feature_name in data.feature_names:
    features[feature_name] = st.number_input(feature_name, value=0.0)

# Prediction button
if st.button("Predict"):
    input_data = np.array(list(features.values())).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    prediction = ann_model.predict(input_data_scaled)
    
    if prediction[0] == 1:
        st.success("The model predicts: **Benign (1)**")
    else:
        st.error("The model predicts: **Malignant (0)**")

# Additional description
st.write("### Model Description")
st.write("This model predicts whether a tumor is malignant or benign based on clinical characteristics.")
# CropNectar – Crop Prediction using Machine Learning and Streamlit

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv("Dataset/Crop_recommendation.csv")

# Split data
X = data.drop('label', axis=1)
y = data['label']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit UI
st.set_page_config(page_title="🌾 CropNectar", layout="centered")

st.title("🌱 CropNectar – AI/ML Based Crop Prediction System")
st.write("Enter the soil and climate details below to predict the best crop to grow.")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0.0)
P = st.number_input("Phosphorus (P)", min_value=0.0)
K = st.number_input("Potassium (K)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)
ph = st.number_input("pH value", min_value=0.0, max_value=14.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

# Predict button
if st.button("🔍 Predict Crop"):
    user_data = [[N, P, K, temperature, humidity, ph, rainfall]]
    prediction = model.predict(user_data)[0]
    st.success(f"🌾 The best crop to grow is: **{prediction.upper()}**")

st.markdown("---")
st.caption("Developed by MCA 3rd Semester Students | School of Computer Science and IT")

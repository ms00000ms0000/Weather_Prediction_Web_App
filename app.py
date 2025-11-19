import streamlit as st
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("weather_model.h5")

st.title("🌦 Weather Prediction using Deep Learning")

st.write("Enter values to predict weather conditions")

precipitation = st.number_input("Precipitation", value=0.0)
temp_max = st.number_input("Maximum Temperature", value=30.0)
temp_min = st.number_input("Minimum Temperature", value=20.0)
wind = st.number_input("Wind Speed", value=5.0)

if st.button("Predict"):
    features = np.array([[precipitation, temp_max, temp_min, wind]])
    prediction = model.predict(features)

    st.success(f"Predicted Value: {prediction[0][0]:.2f}")

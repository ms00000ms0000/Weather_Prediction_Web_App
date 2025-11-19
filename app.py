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

    preds = model.predict(features)

    # step 1: find class index
    class_idx = int(np.argmax(preds))

    # step 2: decode using encoder
    label = encoder.inverse_transform([class_idx])[0]

    st.success(f"Predicted Weather: {label}")
    st.write("Class probabilities:", preds[0])

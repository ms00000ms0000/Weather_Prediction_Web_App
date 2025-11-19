import streamlit as st
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("weather_model.h5")

st.title("🌦 Weather Prediction using Deep Learning")

st.write("Enter values to predict weather conditions")

precipitation = st.number_input("Precipitation", min_value=0.0, max_value=200.0, value=0.0)
temp_max = st.number_input("Maximum Temperature", min_value=-20.0, max_value=60.0, value=30.0)
temp_min = st.number_input("Minimum Temperature", min_value=-30.0, max_value=50.0, value=20.0)
wind = st.number_input("Wind Speed", min_value=0.0, max_value=150.0, value=5.0)

if st.button("Predict"):
    # Prepare input features
    features = np.array([[precipitation, temp_max, temp_min, wind]])

    prediction = model.predict(features)
    class_idx = int(np.argmax(prediction))

    #  5-class mapping
    label_map = {
        0: "drizzle",
        1: "fog",
        2: "rain",
        3: "snow",
        4: "sun"
    }

    label = label_map.get(class_idx, "Unknown")

    st.success(f"Predicted Weather: {label}")

import numpy as np
import tensorflow as tf
import requests
import streamlit as st

# Load trained model
model = tf.keras.models.load_model('fall_detection_model.keras',compile=False)

# Pushbullet API key
PUSHBULLET_TOKEN = 'o.tKCNufo4dhwYfXsHgaW3YHskxXt3iZ9Q'

def send_notification(title, message):
    """Sends notification via Pushbullet."""
    data_send = {"type": "note", "title": title, "body": message}
    headers = {
        'Access-Token': PUSHBULLET_TOKEN,
        'Content-Type': 'application/json'
    }
    response = requests.post('https://api.pushbullet.com/v2/pushes', json=data_send, headers=headers)
    if response.status_code == 200:
        st.success("Notification sent!")
    else:
        st.error(f"Failed to send notification: {response.text}")

def predict_fall(data):
    return model.predict(data.reshape(1, 800), verbose=0)[0][1]

# UI
st.title("📉 Fall Detection Simulator")
st.write("Adjust accelerometer & gyroscope values to simulate activity.")

# Sliders in same scale as dataset
acc_x = st.slider("Acceleration X", -200.0, 200.0, 0.0, step=0.01)
acc_y = st.slider("Acceleration Y", -200.0, 200.0, 0.0, step=0.01)
acc_z = st.slider("Acceleration Z", -200.0, 200.0, 9.81, step=0.01)

gyro_x = st.slider("Gyroscope X", -200.0, 200.0, 0.0, step=0.01)
gyro_y = st.slider("Gyroscope Y", -200.0, 200.0, 0.0, step=0.01)
gyro_z = st.slider("Gyroscope Z", -200.0, 200.0, 0.0, step=0.01)

if st.button("Predict Fall"):
    # Norms exactly like in training
    acc_norm = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    gyro_norm = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

    # Build full 800-length array (100 timesteps × 8 features)
    readings = []
    for _ in range(100):
        readings.extend([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, acc_norm, gyro_norm])

    readings = np.array(readings, dtype=np.float32)

    # Predict
    fall_prob = predict_fall(readings)
    st.metric("Fall Probability", f"{fall_prob:.4f}")

    if fall_prob > 0.5:
        st.error("🚨 Fall Detected!")
        send_notification("Fall Detected", "A possible fall has been detected.")
    else:
        st.success("✅ No Fall Detected.")

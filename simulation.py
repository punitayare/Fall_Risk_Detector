import numpy as np
import tensorflow as tf
import requests

# Load trained model
model = tf.keras.models.load_model('fall_detection_model.keras')

# Pushbullet API key (replace with your actual key)
PUSHBULLET_TOKEN = 'o.tKCNufo4dhwYfXsHgaW3YHskxXt3iZ9Q'

DEVICE_NAME = 'Fall Detector'

def send_notification(title, message):
    """Sends notification via Pushbullet."""
    data_send = {"type": "note", "title": title, "body": message}
    headers = {
        'Access-Token': PUSHBULLET_TOKEN,
        'Content-Type': 'application/json'
    }
    response = requests.post('https://api.pushbullet.com/v2/pushes', json=data_send, headers=headers)
    if response.status_code == 200:
        print("Notification sent!")
    else:
        print(" Failed to send notification:", response.text)
def simulate_sensor_data():
    return np.random.rand(800).astype(np.float32)


def predict_fall(data):
    return model.predict(data.reshape(1, 800), verbose=0)[0][1]

# ---- Main Execution ----
simulated_data = simulate_sensor_data()
fall_probability = predict_fall(simulated_data)

print(f"Fall probability: {fall_probability:.4f}")
if fall_probability > 0.5:
    print("Fall detected!")
    send_notification("Fall Detected", "A possible fall has been detected.")
else:
    print("No fall detected.")

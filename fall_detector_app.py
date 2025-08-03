import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your trained model
model = load_model('fall_detection_model.keras')  # Save your model using model.save('fall_model.h5')

# Title
st.title("🚨 Fall Detection App")
st.write("Upload accelerometer and gyroscope data (.csv) to detect fall or no fall.")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data Preview", df.head())

    try:
        # Calculate norm
        df['acc_norm'] = np.sqrt(df['xAcc']**2 + df['yAcc']**2 + df['zAcc']**2)
        df['gyr_norm'] = np.sqrt(df['xGyro']**2 + df['yGyro']**2 + df['zGyro']**2)

        # Group every 400 rows
        window_size = 400
        if len(df) < window_size:
            st.error("File too small! Must contain at least 400 rows.")
        else:
            acc_norm = df['acc_norm'].values
            gyr_norm = df['gyr_norm'].values
            acc_chunks = [acc_norm[i:i+window_size] for i in range(0, len(acc_norm), window_size) if len(acc_norm[i:i+window_size]) == window_size]
            gyr_chunks = [gyr_norm[i:i+window_size] for i in range(0, len(gyr_norm), window_size) if len(gyr_norm[i:i+window_size]) == window_size]

            predictions = []
            for acc, gyr in zip(acc_chunks, gyr_chunks):
                features = np.hstack((acc, gyr)).reshape(1, -1)  # shape: (1, 800)
                pred = model.predict(features)
                label = np.argmax(pred)
                predictions.append("Fall" if label == 0 else "No Fall")

            st.write("### Predictions for Each Window")
            for i, pred in enumerate(predictions):
                st.write(f"Window {i+1}: **{pred}**")

            # Optional plot
            st.write("### Acceleration Norm Plot")
            plt.figure(figsize=(10, 3))
            plt.plot(df['acc_norm'].values)
            plt.title("Acceleration Norm Over Time")
            plt.xlabel("Sample Index")
            plt.ylabel("acc_norm")
            st.pyplot(plt)

    except Exception as e:
        st.error(f"Error processing the file: {e}")

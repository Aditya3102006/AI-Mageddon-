import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load SCADA dataset (Assumed CSV file format)
data = pd.read_csv("scada_data.csv")

# Feature selection: Selecting relevant SCADA parameters
features = ["power_output", "rotor_speed", "ambient_temperature", "vibration"]
data = data[features]

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Data normalization
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Splitting data into training and testing sets
train_data, test_data = train_test_split(data_scaled, test_size=0.2, random_state=42)

# Autoencoder model for anomaly detection
input_dim = train_data.shape[1]
autoencoder = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
autoencoder.fit(train_data, train_data, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Compute reconstruction loss
reconstructed = autoencoder.predict(test_data)
mse = np.mean(np.power(test_data - reconstructed, 2), axis=1)

# Setting a threshold for anomaly detection
threshold = np.percentile(mse, 95)
anomalies = mse > threshold

# Visualizing anomalies
plt.figure(figsize=(10, 6))
plt.hist(mse, bins=50, alpha=0.75, label='Reconstruction Loss')
plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2, label='Anomaly Threshold')
plt.legend()
plt.xlabel("Mean Squared Error")
plt.ylabel("Frequency")
plt.title("Anomaly Detection in SCADA Data")
plt.show()

# Predictive Maintenance Function
def predict_maintenance(data_point):
    reconstructed = autoencoder.predict([data_point])
    mse = np.mean(np.power(data_point - reconstructed, 2))
    return "Maintenance Required" if mse > threshold else "Normal Operation"

sample_data = test_data[0]  # Taking a sample point
status = predict_maintenance(sample_data)
print("Asset Status:", status)

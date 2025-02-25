import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, regularizers

# Load the CSV file with your 300 rows of data
data = pd.read_csv(r"C:\Users\yarag\OneDrive\Documents\doucments arduino\advanced topic\gesture_recognition_data.csv")

# Inspect the data to understand its structure
print(data.head())

# Separate features (X, Y, Z) and labels
X = data[['X', 'Y', 'Z']].values  # Features (X, Y, Z)
y = data['label'].values          # Labels (Gesture types, e.g., shaking, tilting, still)

# Normalize the data (optional but helps with training)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a deeper neural network model for gesture recognition
model = tf.keras.Sequential([
    layers.InputLayer(input_shape=(3,)),  # 3 input features (X, Y, Z)
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  # First hidden layer with 128 neurons and L2 regularization
    layers.Dropout(0.3),  # Dropout layer with 30% rate
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  # Second hidden layer with 64 neurons
    layers.Dropout(0.3),  # Dropout layer with 30% rate
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  # Third hidden layer with 32 neurons
    layers.Dense(3, activation='softmax')  # 3 output classes (shaking, tilting, still)
])

# Compile the model with a different optimizer and a learning rate scheduler
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Use EarlyStopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Train the model for more epochs (increased from 20 to 50) with validation
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Save the trained model in Keras format
model.save('gesture_recognition_model_improved.keras')

# Convert the model to TensorFlow Lite format for embedded deployment (Arduino)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted TensorFlow Lite model to a file
with open('gesture_recognition_model_improved.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model successfully converted to TensorFlow Lite format and saved as 'gesture_recognition_model_improved.tflite'")

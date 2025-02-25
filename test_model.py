import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Load the trained model (Keras format)
model = tf.keras.models.load_model(r"C:\Users\yarag\OneDrive\Documents\doucments arduino\codes\adv1\gesture_recognition_model_improved.keras")

# Load the CSV file with your data (you can replace this with a new sample for testing)
data = pd.read_csv(r"C:\Users\yarag\OneDrive\Documents\doucments arduino\advanced topic\data 4.csv")

# Inspect the new data (optional, to see the structure)
print(data.head())

# Separate features (X, Y, Z) and labels (gesture type) for the new data
X_new = data[['X', 'Y', 'Z']].values  # Features (X, Y, Z)
y_true = data['label'].values  # True labels (gesture types)

# Normalize the new data using the same scaler used during training
scaler = StandardScaler()
X_new = scaler.fit_transform(X_new)  # Apply normalization to the new data

# Predict the gesture type for the new data
predictions = model.predict(X_new)

# Get the predicted class labels
predicted_classes = np.argmax(predictions, axis=1)

# Display the predictions
print(f"Predictions: {predicted_classes}")

# Calculate accuracy
accuracy = accuracy_score(y_true, predicted_classes)

# Convert to percentage
accuracy_percentage = accuracy * 100

print(f"Accuracy: {accuracy_percentage:.2f}%")

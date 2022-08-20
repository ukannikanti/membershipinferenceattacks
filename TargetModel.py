import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import joblib


# Load the dataset
df = pd.read_csv("./data/diabetes.csv")

# Clean, split the data into test & train
# Use a utility from sklearn to split and shuffle your dataset.
train_df, test_df = train_test_split(df, test_size=0.2)

train_labels = np.array(train_df.pop('Outcome'))
train_labels = train_labels != 0
test_labels = np.array(test_df.pop('Outcome'))

train_features = np.array(train_df)
test_features = np.array(test_df)

# Normalize the data
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)
train_features = np.clip(train_features, -5, 5)
test_features = np.clip(test_features, -5, 5)

print('Training labels shape:', train_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Test features shape:', test_features.shape)

# Build the target model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.fit(train_features, train_labels, epochs=64)

# Evaluate on test dataset
loss, accuracy = model.evaluate(test_features, test_labels)
print("Test Accuracy", accuracy)

# Save the model
joblib.dump(model, 'model.pkl')
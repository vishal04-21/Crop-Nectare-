# Model training script
# Crop Prediction System using Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Step 1: Load the dataset
data = pd.read_csv("Dataset/Crop_recommendation.csv")

# Step 2: Separate features and target
X = data.drop('label', axis=1)  # all columns except 'label'
y = data['label']               # the 'label' column is the crop name

# Step 3: Split the dataset into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5: Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model trained successfully with accuracy: {accuracy * 100:.2f}%")

# Step 6: Take user input for prediction
print("\n Enter the following soil and climate details:")
N = float(input("Nitrogen (N): "))
P = float(input("Phosphorus (P): "))
K = float(input("Potassium (K): "))
temperature = float(input("Temperature (°C): "))
humidity = float(input("Humidity (%): "))
ph = float(input("pH value: "))
rainfall = float(input("Rainfall (mm): "))

# Step 7: Make prediction
user_data = [[N, P, K, temperature, humidity, ph, rainfall]]
predicted_crop = model.predict(user_data)
print(f"\n The best crop to grow is: **{predicted_crop[0].upper()}** 🌾")

import joblib

# Load the trained model
model = joblib.load("electricity_theft_model.pkl")

print("Model loaded successfully!")

import pandas as pd

# Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")

# Prepare input data
X_new = df.drop(columns=["FLAG", "CONS_NO"])

# Take one consumer's data
sample = X_new.iloc[[0]]

# Make prediction
prediction = model.predict(sample)

print("\nPrediction:", prediction[0])

if prediction[0] == 1:
    print("Result: Potentially Suspicious")
else:
    print("Result: Normal")
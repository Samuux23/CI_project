#!/usr/bin/env python3
"""
Test the exact data from the image
"""

import joblib
import pandas as pd

# Load the model and preprocessor
print("Loading model and preprocessor...")
model = joblib.load('best_comprehensive_model.pkl')
preprocessor = joblib.load('preprocessor_comprehensive.pkl')

# Test with the exact data from the image
print("\n" + "="*50)
print("TESTING EXACT DATA FROM IMAGE")
print("="*50)

# Data from the image:
# Time alone: 1 hour
# Social events: 9 per month  
# Going outside: 7 times per week
# Friends: 12 people
# Posts: 6 per week
# Stage fear: No
# Drained: No

exact_data = {
    'Time_spent_Alone': [1.0],  # 1 hour
    'Stage_fear': ['No'],
    'Social_event_attendance': [9],  # 9 per month
    'Going_outside': [7],  # 7 times per week
    'Drained_after_socializing': ['No'],
    'Friends_circle_size': [12],  # 12 people
    'Post_frequency': [6]  # 6 per week
}

print(f"Input data: {exact_data}")

# Create DataFrame
input_df = pd.DataFrame(exact_data)
print(f"DataFrame shape: {input_df.shape}")
print(f"DataFrame columns: {input_df.columns.tolist()}")

# Preprocess
processed_data = preprocessor.transform(input_df)
print(f"Processed data shape: {processed_data.shape}")

# Make prediction
prediction_proba = model.predict_proba(processed_data)[0]
prediction = model.predict(processed_data)[0]

print(f"Raw prediction: {prediction}")
print(f"Prediction probabilities: {prediction_proba}")
print(f"Class 0 probability: {prediction_proba[0]:.4f}")
print(f"Class 1 probability: {prediction_proba[1]:.4f}")

# Map prediction correctly
personality_type = "Extrovert" if prediction == 0 else "Introvert"
confidence = prediction_proba[prediction]

print(f"Personality type: {personality_type}")
print(f"Confidence: {confidence:.4f}")

# Check if this matches what the app should show
print(f"\nExpected result: {personality_type} with {confidence*100:.2f}% confidence") 
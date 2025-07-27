#!/usr/bin/env python3
"""
Debug script to test the personality prediction model
"""

import joblib
import pandas as pd
import numpy as np

# Load the model and preprocessor
print("Loading model and preprocessor...")
model = joblib.load('best_comprehensive_model.pkl')
preprocessor = joblib.load('preprocessor_comprehensive.pkl')

# Try to load label encoder if it exists
try:
    label_encoder = joblib.load('label_encoder_comprehensive.pkl')
    print(f"Label encoder classes: {label_encoder.classes_}")
    print(f"Label encoder mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
except:
    print("No label encoder found")

# Test with extrovert pattern
print("\n" + "="*50)
print("TESTING EXTROVERT PATTERN")
print("="*50)

extrovert_data = {
    'Time_spent_Alone': [1.5],  # 1-2 hours
    'Stage_fear': ['No'],
    'Social_event_attendance': [10],  # 8+ per month
    'Going_outside': [7],  # 6+ times per week
    'Drained_after_socializing': ['No'],
    'Friends_circle_size': [15],  # 12+ people
    'Post_frequency': [6]  # 5+ per week
}

print(f"Input data: {extrovert_data}")

# Create DataFrame
input_df = pd.DataFrame(extrovert_data)
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

# Check what the classes mean
print(f"\nModel classes: {model.classes_}")
print(f"Number of classes: {len(model.classes_)}")

# Map prediction
if len(model.classes_) == 2:
    personality_type = model.classes_[prediction]
    confidence = prediction_proba[prediction]
    print(f"Personality type: {personality_type}")
    print(f"Confidence: {confidence:.4f}")
else:
    print("Unexpected number of classes!")

# Test with introvert pattern
print("\n" + "="*50)
print("TESTING INTROVERT PATTERN")
print("="*50)

introvert_data = {
    'Time_spent_Alone': [7.0],  # 6+ hours
    'Stage_fear': ['Yes'],
    'Social_event_attendance': [2],  # 1-2 per month
    'Going_outside': [2],  # 1-2 times per week
    'Drained_after_socializing': ['Yes'],
    'Friends_circle_size': [4],  # 3-5 people
    'Post_frequency': [1]  # 0-2 per week
}

print(f"Input data: {introvert_data}")

# Create DataFrame
input_df = pd.DataFrame(introvert_data)
processed_data = preprocessor.transform(input_df)

# Make prediction
prediction_proba = model.predict_proba(processed_data)[0]
prediction = model.predict(processed_data)[0]

print(f"Raw prediction: {prediction}")
print(f"Prediction probabilities: {prediction_proba}")
print(f"Class 0 probability: {prediction_proba[0]:.4f}")
print(f"Class 1 probability: {prediction_proba[1]:.4f}")

# Map prediction
if len(model.classes_) == 2:
    personality_type = model.classes_[prediction]
    confidence = prediction_proba[prediction]
    print(f"Personality type: {personality_type}")
    print(f"Confidence: {confidence:.4f}")

print("\n" + "="*50)
print("MODEL INFO")
print("="*50)
print(f"Model type: {type(model).__name__}")
print(f"Model classes: {model.classes_}")
print(f"Feature names: {getattr(model, 'feature_names_in_', 'Not available')}")

# Test with some training data samples
print("\n" + "="*50)
print("TESTING WITH TRAINING DATA SAMPLES")
print("="*50)

# Load some training data
train_df = pd.read_csv('train.csv')
print(f"Training data shape: {train_df.shape}")

# Test a few extrovert samples
extrovert_samples = train_df[train_df['Personality'] == 'Extrovert'].head(3)
print(f"\nExtrovert samples from training data:")
for idx, row in extrovert_samples.iterrows():
    sample_data = {
        'Time_spent_Alone': [row['Time_spent_Alone']],
        'Stage_fear': [row['Stage_fear']],
        'Social_event_attendance': [row['Social_event_attendance']],
        'Going_outside': [row['Going_outside']],
        'Drained_after_socializing': [row['Drained_after_socializing']],
        'Friends_circle_size': [row['Friends_circle_size']],
        'Post_frequency': [row['Post_frequency']]
    }
    
    input_df = pd.DataFrame(sample_data)
    processed_data = preprocessor.transform(input_df)
    prediction_proba = model.predict_proba(processed_data)[0]
    prediction = model.predict(processed_data)[0]
    
    print(f"Sample {idx}: Expected=Extrovert, Predicted=Class{prediction}, Proba={prediction_proba}")

# Test a few introvert samples
introvert_samples = train_df[train_df['Personality'] == 'Introvert'].head(3)
print(f"\nIntrovert samples from training data:")
for idx, row in introvert_samples.iterrows():
    sample_data = {
        'Time_spent_Alone': [row['Time_spent_Alone']],
        'Stage_fear': [row['Stage_fear']],
        'Social_event_attendance': [row['Social_event_attendance']],
        'Going_outside': [row['Going_outside']],
        'Drained_after_socializing': [row['Drained_after_socializing']],
        'Friends_circle_size': [row['Friends_circle_size']],
        'Post_frequency': [row['Post_frequency']]
    }
    
    input_df = pd.DataFrame(sample_data)
    processed_data = preprocessor.transform(input_df)
    prediction_proba = model.predict_proba(processed_data)[0]
    prediction = model.predict(processed_data)[0]
    
    print(f"Sample {idx}: Expected=Introvert, Predicted=Class{prediction}, Proba={prediction_proba}") 
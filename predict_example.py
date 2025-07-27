import joblib
import pandas as pd
import numpy as np

def load_model():
    """Load the trained model and preprocessor"""
    model = joblib.load('best_personality_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    feature_names = joblib.load('feature_names.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    
    return model, preprocessor, feature_names, label_encoder

def predict_personality(data, model, preprocessor, label_encoder):
    """Predict personality for new data"""
    # Preprocess the data
    X_processed = preprocessor.transform(data)
    
    # Make predictions
    predictions_encoded = model.predict(X_processed)
    
    # Convert back to original labels
    predictions = label_encoder.inverse_transform(predictions_encoded)
    
    # Get prediction probabilities
    probabilities = model.predict_proba(X_processed)
    
    return predictions, probabilities

def main():
    """Example usage of the trained model"""
    print("Loading the trained personality classification model...")
    
    try:
        model, preprocessor, feature_names, label_encoder = load_model()
        print("Model loaded successfully!")
        
        # Example 1: Predict for a single person
        print("\n=== Example 1: Single Person Prediction ===")
        
        # Create sample data for one person
        sample_data = pd.DataFrame({
            'Time_spent_Alone': [2.0],
            'Stage_fear': ['No'],
            'Social_event_attendance': [6.0],
            'Going_outside': [4.0],
            'Drained_after_socializing': ['No'],
            'Friends_circle_size': [10.0],
            'Post_frequency': [5.0]
        })
        
        print("Sample person data:")
        print(sample_data)
        
        prediction, probability = predict_personality(sample_data, model, preprocessor, label_encoder)
        
        print(f"\nPredicted Personality: {prediction[0]}")
        print(f"Confidence: {max(probability[0])*100:.1f}%")
        print(f"Probability breakdown:")
        for i, (label, prob) in enumerate(zip(label_encoder.classes_, probability[0])):
            print(f"  {label}: {prob*100:.1f}%")
        
        # Example 2: Predict for multiple people
        print("\n=== Example 2: Multiple People Prediction ===")
        
        multiple_data = pd.DataFrame({
            'Time_spent_Alone': [8.0, 1.0, 5.0],
            'Stage_fear': ['Yes', 'No', 'Yes'],
            'Social_event_attendance': [2.0, 8.0, 3.0],
            'Going_outside': [1.0, 6.0, 2.0],
            'Drained_after_socializing': ['Yes', 'No', 'Yes'],
            'Friends_circle_size': [3.0, 12.0, 5.0],
            'Post_frequency': [1.0, 7.0, 2.0]
        })
        
        print("Multiple people data:")
        print(multiple_data)
        
        predictions, probabilities = predict_personality(multiple_data, model, preprocessor, label_encoder)
        
        print("\nPredictions:")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            confidence = max(prob) * 100
            print(f"Person {i+1}: {pred} (Confidence: {confidence:.1f}%)")
        
        # Example 3: Model information
        print("\n=== Example 3: Model Information ===")
        print(f"Model type: {type(model).__name__}")
        print(f"Number of features: {len(feature_names)}")
        print(f"Feature names: {feature_names}")
        print(f"Classes: {label_encoder.classes_}")
        
        if hasattr(model, 'feature_importances_'):
            print("\nFeature importances:")
            importances = model.feature_importances_
            for feature, importance in zip(feature_names, importances):
                print(f"  {feature}: {importance:.4f}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you have run the main pipeline first to generate the model files.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 
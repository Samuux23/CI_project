from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# Initialize FastAPI app
app = FastAPI(title="Personality Predictor API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and preprocessor
try:
    model = joblib.load('../best_comprehensive_model.pkl')
    preprocessor = joblib.load('../preprocessor_comprehensive.pkl')
    print("✅ Model and preprocessor loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    preprocessor = None

# Define the input model
class PersonalityInput(BaseModel):
    time_spent_alone: float
    social_event_attendance: float
    going_outside: float
    friends_circle_size: float
    post_frequency: float
    stage_fear: str  # "Yes" or "No"
    drained_after_socializing: str  # "Yes" or "No"

# Define the response model
class PersonalityResponse(BaseModel):
    prediction: str
    confidence: float
    personality_type: str

@app.get("/")
async def root():
    return {"message": "Personality Predictor API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PersonalityResponse)
async def predict_personality(input_data: PersonalityInput):
    try:
        if model is None or preprocessor is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Create input DataFrame
        input_dict = {
            'Time_spent_Alone': [input_data.time_spent_alone],
            'Social_event_attendance': [input_data.social_event_attendance],
            'Going_outside': [input_data.going_outside],
            'Friends_circle_size': [input_data.friends_circle_size],
            'Post_frequency': [input_data.post_frequency],
            'Stage_fear': [input_data.stage_fear],
            'Drained_after_socializing': [input_data.drained_after_socializing]
        }
        
        input_df = pd.DataFrame(input_dict)
        
        # Preprocess the input data
        processed_data = preprocessor.transform(input_df)
        
        # Make prediction
        prediction_proba = model.predict_proba(processed_data)[0]
        prediction = model.predict(processed_data)[0]
        
        # Get confidence (probability of the predicted class)
        confidence = max(prediction_proba)
        
        # Map prediction to personality type
        personality_type = "Introvert" if prediction == 0 else "Extrovert"
        
        return PersonalityResponse(
            prediction=personality_type,
            confidence=round(confidence * 100, 2),
            personality_type=personality_type
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
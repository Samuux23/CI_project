#!/usr/bin/env python3
"""
Personality Predictor - Single File Application
Combines frontend and backend in one file for easy deployment
"""

import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model and preprocessor
try:
    model = joblib.load('best_comprehensive_model.pkl')
    preprocessor = joblib.load('preprocessor_comprehensive.pkl')
    print("✅ Model and preprocessor loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    preprocessor = None

# HTML template for the frontend
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Personality Predictor - AI Model</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 20px;
            background: #f8f9fa;
        }

        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .stat-number {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }

        .stat-label {
            color: #666;
            margin-top: 5px;
        }

        .form-container {
            padding: 30px;
        }

        .form-group {
            margin-bottom: 20px;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }

        .form-group input, .form-group select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }

        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .radio-group {
            display: flex;
            gap: 20px;
        }

        .radio-option {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .submit-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            transition: transform 0.2s ease;
        }

        .submit-btn:hover {
            transform: translateY(-2px);
        }

        .submit-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .result-container {
            padding: 30px;
            text-align: center;
            display: none;
        }

        .result-title {
            font-size: 2rem;
            margin-bottom: 15px;
            color: #333;
        }

        .personality-type {
            font-size: 3rem;
            font-weight: bold;
            margin: 20px 0;
            padding: 20px;
            border-radius: 10px;
        }

        .introvert {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .extrovert {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }

        .loading {
            text-align: center;
            padding: 40px;
            display: none;
        }

        .spinner {
            border: 4px solid rgba(102, 126, 234, 0.3);
            border-radius: 50%;
            border-top: 4px solid #667eea;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .error {
            background: #fee;
            color: #c33;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #fcc;
            margin-bottom: 20px;
            display: none;
        }

        .reset-btn {
            background: #6c757d;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            margin-top: 15px;
        }

        .reset-btn:hover {
            background: #5a6268;
        }

        .confidence-bar {
            width: 100%;
            height: 20px;
            background: #e1e5e9;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }

        .confidence-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            transition: width 1s ease;
        }

        .footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #e1e5e9;
        }

        @media (max-width: 600px) {
            .header h1 {
                font-size: 2rem;
            }
            
            .radio-group {
                flex-direction: column;
                gap: 10px;
            }
            
            .stats {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Personality Predictor</h1>
            <p>AI-Powered Personality Analysis with 96.9% Accuracy</p>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">96.9%</div>
                <div class="stat-label">Model Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">7</div>
                <div class="stat-label">Input Features</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">2</div>
                <div class="stat-label">Personality Types</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">18K+</div>
                <div class="stat-label">Training Samples</div>
            </div>
        </div>

        <div class="form-container" id="formContainer">
            <h2 style="margin-bottom: 20px; color: #333;">Enter Your Behavioral Data</h2>
            
            <div class="error" id="errorMessage"></div>

            <form id="personalityForm">
                <div class="form-group">
                    <label>Time Spent Alone (hours per day)</label>
                    <input type="number" name="time_spent_alone" placeholder="e.g., 4" min="0" step="0.1" required>
                </div>

                <div class="form-group">
                    <label>Social Event Attendance (per month)</label>
                    <input type="number" name="social_event_attendance" placeholder="e.g., 8" min="0" required>
                </div>

                <div class="form-group">
                    <label>Going Outside (times per week)</label>
                    <input type="number" name="going_outside" placeholder="e.g., 5" min="0" required>
                </div>

                <div class="form-group">
                    <label>Friends Circle Size</label>
                    <input type="number" name="friends_circle_size" placeholder="e.g., 10" min="0" required>
                </div>

                <div class="form-group">
                    <label>Post Frequency (per week)</label>
                    <input type="number" name="post_frequency" placeholder="e.g., 3" min="0" required>
                </div>

                <div class="form-group">
                    <label>Stage Fear</label>
                    <div class="radio-group">
                        <label class="radio-option">
                            <input type="radio" name="stage_fear" value="Yes" required>
                            Yes
                        </label>
                        <label class="radio-option">
                            <input type="radio" name="stage_fear" value="No" required>
                            No
                        </label>
                    </div>
                </div>

                <div class="form-group">
                    <label>Drained After Socializing</label>
                    <div class="radio-group">
                        <label class="radio-option">
                            <input type="radio" name="drained_after_socializing" value="Yes" required>
                            Yes
                        </label>
                        <label class="radio-option">
                            <input type="radio" name="drained_after_socializing" value="No" required>
                            No
                        </label>
                    </div>
                </div>

                <button type="submit" class="submit-btn" id="submitBtn">
                    🚀 Get My Personality Prediction
                </button>
            </form>
        </div>

        <div class="loading" id="loadingContainer">
            <div class="spinner"></div>
            <p>Analyzing your personality patterns...</p>
        </div>

        <div class="result-container" id="resultContainer">
            <h2 class="result-title">Your Personality Result</h2>
            <p id="confidenceText" style="font-size: 1.2rem; color: #666; margin-bottom: 20px;"></p>
            <div class="confidence-bar">
                <div class="confidence-fill" id="confidenceBar"></div>
            </div>
            <div class="personality-type" id="personalityType"></div>
            <p id="resultDescription" style="color: #666; margin-bottom: 20px;"></p>
            <button onclick="resetForm()" class="reset-btn">🔄 Test Again</button>
        </div>

        <div class="footer">
            <p>Powered by Machine Learning • Built with Flask • 96.9% Accuracy</p>
        </div>
    </div>

    <script>
        const form = document.getElementById('personalityForm');
        const formContainer = document.getElementById('formContainer');
        const loadingContainer = document.getElementById('loadingContainer');
        const resultContainer = document.getElementById('resultContainer');
        const errorMessage = document.getElementById('errorMessage');
        const submitBtn = document.getElementById('submitBtn');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // Show loading
            formContainer.style.display = 'none';
            loadingContainer.style.display = 'block';
            resultContainer.style.display = 'none';
            errorMessage.style.display = 'none';
            submitBtn.disabled = true;

            // Get form data
            const formData = new FormData(form);
            const data = {
                time_spent_alone: parseFloat(formData.get('time_spent_alone')),
                social_event_attendance: parseFloat(formData.get('social_event_attendance')),
                going_outside: parseFloat(formData.get('going_outside')),
                friends_circle_size: parseFloat(formData.get('friends_circle_size')),
                post_frequency: parseFloat(formData.get('post_frequency')),
                stage_fear: formData.get('stage_fear'),
                drained_after_socializing: formData.get('drained_after_socializing')
            };

            try {
                // Call the API
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data),
                });

                if (!response.ok) {
                    throw new Error('Failed to get prediction');
                }

                const result = await response.json();
                showResult(result);
            } catch (err) {
                showError('Failed to get prediction. Please try again.');
                console.error('Prediction error:', err);
            } finally {
                submitBtn.disabled = false;
            }
        });

        function showResult(result) {
            loadingContainer.style.display = 'none';
            resultContainer.style.display = 'block';

            const personalityType = document.getElementById('personalityType');
            const confidenceText = document.getElementById('confidenceText');
            const confidenceBar = document.getElementById('confidenceBar');
            const resultDescription = document.getElementById('resultDescription');

            personalityType.textContent = result.personality_type;
            personalityType.className = `personality-type ${result.personality_type.toLowerCase()}`;
            
            confidenceText.textContent = `Confidence: ${result.confidence}%`;
            confidenceBar.style.width = `${result.confidence}%`;
            
            resultDescription.textContent = `Based on your behavioral patterns, you are likely an ${result.personality_type.toLowerCase()}.`;
        }

        function showError(message) {
            loadingContainer.style.display = 'none';
            formContainer.style.display = 'block';
            errorMessage.textContent = message;
            errorMessage.style.display = 'block';
        }

        function resetForm() {
            form.reset();
            formContainer.style.display = 'block';
            resultContainer.style.display = 'none';
            errorMessage.style.display = 'none';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Serve the main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for personality prediction"""
    try:
        if model is None or preprocessor is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get input data
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'time_spent_alone', 'social_event_attendance', 'going_outside',
            'friends_circle_size', 'post_frequency', 'stage_fear', 'drained_after_socializing'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create input DataFrame with correct column names matching training data
        input_dict = {
            'Time_spent_Alone': [data['time_spent_alone']],
            'Stage_fear': [data['stage_fear']],
            'Social_event_attendance': [data['social_event_attendance']],
            'Going_outside': [data['going_outside']],
            'Drained_after_socializing': [data['drained_after_socializing']],
            'Friends_circle_size': [data['friends_circle_size']],
            'Post_frequency': [data['post_frequency']]
        }
        
        input_df = pd.DataFrame(input_dict)
        
        # Preprocess the input data
        processed_data = preprocessor.transform(input_df)
        
        # Make prediction
        prediction_proba = model.predict_proba(processed_data)[0]
        prediction = model.predict(processed_data)[0]
        
        # Get confidence (probability of the predicted class)
        confidence = max(prediction_proba)
        
        # Map prediction to personality type using label encoder
        # Class 0 = Extrovert, Class 1 = Introvert (based on label encoder)
        personality_type = "Extrovert" if prediction == 0 else "Introvert"
        
        # Debug logging
        print(f"Input data: {input_dict}")
        print(f"Prediction: {prediction}")
        print(f"Prediction probabilities: {prediction_proba}")
        print(f"Personality type: {personality_type}")
        print(f"Confidence: {confidence}")
        
        return jsonify({
            'prediction': personality_type,
            'confidence': round(confidence * 100, 2),
            'personality_type': personality_type
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'accuracy': '96.9%'
    })

if __name__ == '__main__':
    print("🧠 Personality Predictor - Single File Application")
    print("=" * 50)
    print("🚀 Starting server...")
    print("📱 Frontend: http://localhost:5000")
    print("🔧 API: http://localhost:5000/predict")
    print("🏥 Health: http://localhost:5000/health")
    print("=" * 50)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False) 
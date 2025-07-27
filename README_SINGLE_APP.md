# 🧠 Personality Predictor - Single File Application

A complete personality prediction application combining frontend and backend in a single Python file for easy deployment and publishing.

## ✨ Features

- **🎯 Single File**: Everything in one `personality_predictor_app.py` file
- **🤖 AI Model**: 96.9% accurate personality prediction
- **🎨 Beautiful UI**: Modern, responsive design with gradients
- **📱 Mobile Friendly**: Works on all devices
- **⚡ Fast**: Real-time predictions
- **🔒 Secure**: Input validation and error handling
- **📊 Statistics**: Model accuracy and feature information
- **🎯 Easy Deployment**: One command to run

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Required model files (included in project)

### Installation & Run

#### Option 1: Simple Start (Windows)
```bash
# Double-click or run:
start_app.bat
```

#### Option 2: Manual Start
```bash
# Install dependencies
pip install -r requirements_single_app.txt

# Run the application
python personality_predictor_app.py
```

#### Option 3: Direct Run (if dependencies already installed)
```bash
python personality_predictor_app.py
```

## 🌐 Access Points

Once running, access the application at:

- **📱 Main Application**: http://localhost:5000
- **🔧 API Endpoint**: http://localhost:5000/predict
- **🏥 Health Check**: http://localhost:5000/health

## 📋 Input Fields

The application collects 7 behavioral patterns:

1. **Time Spent Alone** (hours per day)
2. **Social Event Attendance** (per month)
3. **Going Outside** (times per week)
4. **Friends Circle Size**
5. **Post Frequency** (per week)
6. **Stage Fear** (Yes/No)
7. **Drained After Socializing** (Yes/No)

## 🎯 Output

- **Personality Type**: Introvert or Extrovert
- **Confidence Score**: Percentage confidence in prediction
- **Visual Feedback**: Color-coded results with confidence bar

## 🏗️ Architecture

### Single File Structure
```
personality_predictor_app.py
├── Flask App Setup
├── Model Loading
├── HTML Template (Frontend)
├── API Routes
│   ├── / (Main Page)
│   ├── /predict (Prediction API)
│   └── /health (Health Check)
└── Server Startup
```

### Technologies Used
- **Backend**: Flask, scikit-learn, pandas, numpy
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Model**: Gradient Boosting Classifier (96.9% accuracy)
- **Styling**: Custom CSS with gradients and animations

## 📦 Dependencies

```
flask==3.1.1
flask-cors==6.0.1
joblib==1.5.1
pandas==2.3.1
numpy==2.2.6
scikit-learn==1.6.1
```

## 🚀 Deployment Options

### 1. Local Development
```bash
python personality_predictor_app.py
```

### 2. Production Server
```bash
# Using gunicorn (Linux/Mac)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 personality_predictor_app:app

# Using waitress (Windows)
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 personality_predictor_app:app
```

### 3. Cloud Platforms

#### Heroku
```bash
# Create Procfile
echo "web: gunicorn personality_predictor_app:app" > Procfile

# Deploy
git add .
git commit -m "Deploy personality predictor"
heroku create your-app-name
git push heroku main
```

#### Railway
```bash
# Connect your GitHub repo
# Railway will auto-detect Flask app
```

#### Render
```bash
# Connect your GitHub repo
# Set build command: pip install -r requirements_single_app.txt
# Set start command: gunicorn personality_predictor_app:app
```

#### Vercel
```bash
# Create vercel.json
{
  "version": 2,
  "builds": [
    {
      "src": "personality_predictor_app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "personality_predictor_app.py"
    }
  ]
}
```

## 🔧 API Documentation

### POST /predict
Predict personality type from behavioral data.

**Request Body:**
```json
{
  "time_spent_alone": 4.5,
  "social_event_attendance": 8,
  "going_outside": 5,
  "friends_circle_size": 10,
  "post_frequency": 3,
  "stage_fear": "Yes",
  "drained_after_socializing": "Yes"
}
```

**Response:**
```json
{
  "prediction": "Introvert",
  "confidence": 87.5,
  "personality_type": "Introvert"
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "accuracy": "96.9%"
}
```

## 📁 File Structure

```
Task CI/
├── personality_predictor_app.py      # Main application file
├── requirements_single_app.txt       # Dependencies
├── start_app.bat                    # Windows startup script
├── README_SINGLE_APP.md             # This file
├── best_comprehensive_model.pkl     # Trained model
├── preprocessor_comprehensive.pkl   # Data preprocessor
└── [other ML files...]
```

## 🎨 Customization

### Changing Colors
Edit the CSS variables in the HTML template:
```css
/* Primary colors */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* Introvert colors */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* Extrovert colors */
background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
```

### Adding Features
- Modify the HTML template for UI changes
- Add new routes in the Flask app
- Extend the prediction logic

## 🔍 Troubleshooting

### Common Issues

1. **Port 5000 already in use**
   ```bash
   # Change port in the file
   app.run(host='0.0.0.0', port=5001, debug=False)
   ```

2. **Model files not found**
   - Ensure `best_comprehensive_model.pkl` and `preprocessor_comprehensive.pkl` are in the same directory

3. **Dependencies not installed**
   ```bash
   pip install -r requirements_single_app.txt
   ```

4. **CORS issues**
   - The app includes CORS middleware, but you can modify if needed

### Debug Mode
```python
# Change debug=True for development
app.run(host='0.0.0.0', port=5000, debug=True)
```

## 📊 Model Information

- **Algorithm**: Gradient Boosting Classifier
- **Accuracy**: 96.9%
- **Training Data**: 18,526 samples
- **Features**: 7 behavioral patterns
- **Classes**: Introvert (0), Extrovert (1)

## 🤝 Contributing

1. Fork the repository
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

If you encounter any issues:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure model files are present
4. Check the console for error messages

---

**🎉 Enjoy your single-file personality predictor application!** 
# 🧠 Personality Predictor - Test Application

A simple frontend application to test your trained personality prediction model.

## 🚀 Quick Start

### Option 1: Automated Startup (Recommended)
```bash
python start_test_app.py
```

### Option 2: Manual Startup

#### Step 1: Start Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```

#### Step 2: Start Frontend (in new terminal)
```bash
cd frontend
npm install
npm start
```

## 🌐 Access Points

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📋 Features

- ✅ **Simple Form Interface** with all 7 input fields
- ✅ **Real Model Integration** using your trained model
- ✅ **Instant Predictions** with confidence scores
- ✅ **Beautiful UI** with modern styling
- ✅ **Mobile Responsive** design
- ✅ **Error Handling** for better UX

## 🎯 How to Test

1. **Fill out the form** with your behavioral patterns:
   - Time spent alone (hours per day)
   - Social event attendance (per month)
   - Going outside (times per week)
   - Friends circle size
   - Post frequency (per week)
   - Stage fear (Yes/No)
   - Drained after socializing (Yes/No)

2. **Click "Get Personality Prediction"** to receive your result

3. **View your result** with confidence score and personality type

4. **Click "Test Again"** to try with different inputs

## 🔧 Technical Details

- **Frontend**: React with modern CSS
- **Backend**: FastAPI with your trained model
- **Model**: Uses `best_comprehensive_model.pkl` and `preprocessor_comprehensive.pkl`
- **API**: RESTful endpoint at `/predict`

## 📁 File Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── App.js          # Main React component
│   ├── index.js        # React entry point
│   └── index.css       # Styling
└── package.json        # Dependencies

backend/
├── main.py             # FastAPI server
└── requirements.txt    # Python dependencies

start_test_app.py       # Startup script
```

## 🐛 Troubleshooting

### Common Issues:
1. **Port conflicts**: Make sure ports 3000 and 8000 are available
2. **Model loading**: Ensure model files are in the main directory
3. **Dependencies**: Run `pip install` and `npm install` if needed

### Manual Testing:
```bash
# Test backend directly
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "time_spent_alone": 4,
    "social_event_attendance": 6,
    "going_outside": 5,
    "friends_circle_size": 10,
    "post_frequency": 3,
    "stage_fear": "No",
    "drained_after_socializing": "No"
  }'
```

## 🎉 Ready to Test!

The application is now ready to test your personality prediction model. Simply run the startup script and start making predictions! 
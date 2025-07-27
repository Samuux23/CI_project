@echo off
echo 🧠 Personality Predictor - Single File Application
echo ================================================
echo.
echo 📦 Installing dependencies...
pip install -r requirements_single_app.txt
echo.
echo 🚀 Starting the application...
echo 📱 Frontend: http://localhost:5000
echo 🔧 API: http://localhost:5000/predict
echo 🏥 Health: http://localhost:5000/health
echo.
echo Press Ctrl+C to stop the server
echo.
python personality_predictor_app.py
pause 
# 🧠 Personality Predictor - AI Model

A machine learning application that predicts personality types (Introvert/Extrovert) based on behavioral patterns with **96.9% accuracy**.

## 🌐 Live Application

**[🚀 Try the Live App](https://samuux23.github.io/CI_project/)**

> **Note**: The web application is now live at the root URL above. Click the link to access the interactive personality predictor!

## ✨ Features

- **🤖 AI-Powered**: 96.9% accurate personality prediction
- **📊 Beautiful UI**: Modern, responsive design
- **📱 Mobile Friendly**: Works on all devices
- **⚡ Real-time**: Instant predictions with confidence scores
- **📈 Insights**: Detailed personality trait explanations
- **🔒 Privacy**: No data storage, all processing is local

## 🎯 How It Works

The application analyzes 7 key behavioral patterns:

1. **Time Spent Alone** (hours per day)
2. **Social Event Attendance** (per month)
3. **Going Outside** (times per week)
4. **Friends Circle Size**
5. **Post Frequency** (per week)
6. **Stage Fear** (Yes/No)
7. **Drained After Socializing** (Yes/No)

Based on these inputs, the AI model predicts whether you're an **Introvert** or **Extrovert** with a confidence score.

## 🚀 Quick Start

### Option 1: Use the Live App
Simply visit **[https://samuux23.github.io/CI_project/](https://samuux23.github.io/CI_project/)** and start predicting!

### Option 2: Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/Samuux23/CI_project.git
   cd CI_project
   ```

2. **Open index.html in browser**
   - Simply open `index.html` in your web browser
   - No server setup required!

3. **Alternative: Run Streamlit version**
   ```bash
   pip install -r requirements.txt
   streamlit run streamlit_app.py
   ```

## 📊 Model Performance

- **Algorithm**: Gradient Boosting Classifier
- **Accuracy**: 96.9%
- **Training Data**: 18,526 samples
- **Features**: 7 behavioral patterns
- **Classes**: Introvert (0), Extrovert (1)

## 🏗️ Project Structure

```
CI_project/
├── index.html                    # Static web app (GitHub Pages)
├── streamlit_app.py              # Streamlit application
├── personality_predictor_app.py  # Flask version (single file)
├── requirements.txt              # Python dependencies
├── best_comprehensive_model.pkl  # Trained model
├── preprocessor_comprehensive.pkl # Data preprocessor
├── train.csv                     # Training dataset
├── test.csv                      # Test dataset
├── README.md                     # This file
└── .github/workflows/            # GitHub Actions
    └── streamlit-deploy.yml      # Deployment workflow
```

## 🎨 Technologies Used

- **Frontend**: HTML, CSS, JavaScript (GitHub Pages)
- **Alternative**: Streamlit, Python, scikit-learn
- **ML Model**: Gradient Boosting Classifier
- **Deployment**: GitHub Pages, Streamlit Cloud
- **Styling**: Custom CSS with gradients

## 📈 Model Training

The model was trained using:
- **Hyperparameter Optimization**: Comprehensive grid search
- **Feature Engineering**: Advanced preprocessing pipeline
- **Cross-validation**: 5-fold CV for robust evaluation
- **Ensemble Methods**: Multiple algorithms tested

## 🔧 Development

### Static Version (Recommended for GitHub Pages)
The `index.html` file contains a complete static version of the personality predictor that:
- ✅ Works without any server setup
- ✅ Runs entirely in the browser
- ✅ Uses JavaScript-based prediction logic
- ✅ Maintains the same beautiful UI
- ✅ Provides instant results

### Streamlit Version
For the full ML model experience, use `streamlit_app.py` which:
- ✅ Uses the actual trained ML model
- ✅ Provides more accurate predictions
- ✅ Requires Python environment setup

## 🚀 Deployment

### GitHub Pages (Current)
- **URL**: https://samuux23.github.io/CI_project/
- **Type**: Static HTML/CSS/JavaScript
- **Status**: ✅ Live and working

### Streamlit Cloud (Alternative)
- **URL**: https://personality-predictor.streamlit.app
- **Type**: Full ML model deployment
- **Status**: Available for advanced users

## 📝 Usage Examples

### Extrovert Pattern
- Time alone: 1-2 hours
- Social events: 8+ per month
- Going outside: 6+ times per week
- Friends: 12+ people
- Posts: 5+ per week
- Stage fear: No
- Drained: No

### Introvert Pattern
- Time alone: 6+ hours
- Social events: 2-3 per month
- Going outside: 2-3 times per week
- Friends: 2-5 people
- Posts: 1-2 per week
- Stage fear: Yes
- Drained: Yes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Built with modern web technologies
- Powered by machine learning algorithms
- Deployed on GitHub Pages for easy access

---

**🎯 Ready to discover your personality type? [Click here to start!](https://samuux23.github.io/CI_project/)** 
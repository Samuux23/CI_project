# 🧠 Personality Predictor - AI Model

A machine learning application that predicts personality types (Introvert/Extrovert) based on behavioral patterns with **96.9% accuracy**.

## 🌐 Live Application

**[🚀 Try the Live App](https://personality-predictor.streamlit.app)**

## ✨ Features

- **🤖 AI-Powered**: 96.9% accurate personality prediction
- **📊 Beautiful UI**: Modern, responsive design with Streamlit
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
Simply visit **[https://personality-predictor.streamlit.app](https://personality-predictor.streamlit.app)** and start predicting!

### Option 2: Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/Samuux23/CI_project.git
   cd CI_project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open in browser**
   - Navigate to `http://localhost:8501`
   - Start predicting personalities!

## 📊 Model Performance

- **Algorithm**: Gradient Boosting Classifier
- **Accuracy**: 96.9%
- **Training Data**: 18,526 samples
- **Features**: 7 behavioral patterns
- **Classes**: Introvert (0), Extrovert (1)

## 🏗️ Project Structure

```
CI_project/
├── streamlit_app.py              # Main Streamlit application
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

- **Frontend**: Streamlit
- **Backend**: Python, scikit-learn
- **ML Model**: Gradient Boosting Classifier
- **Deployment**: Streamlit Cloud, GitHub Actions
- **Styling**: Custom CSS with gradients

## 📈 Model Training

The model was trained using:
- **Hyperparameter Optimization**: Comprehensive grid search
- **Feature Engineering**: Advanced preprocessing pipeline
- **Cross-validation**: 5-fold CV for robust evaluation
- **Ensemble Methods**: Multiple algorithms tested

## 🔧 Development

### Local Development
```bash
# Install development dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py

# Run tests
python test_model_debug.py
```

### Deployment
The app is automatically deployed to Streamlit Cloud when you push to the main branch.

## 📱 Usage Examples

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
- Social events: 1-2 per month
- Going outside: 1-2 times per week
- Friends: 3-5 people
- Posts: 0-2 per week
- Stage fear: Yes
- Drained: Yes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: Personality prediction dataset
- Streamlit: For the amazing web app framework
- scikit-learn: For the machine learning tools
- GitHub: For hosting and deployment

## 📞 Support

If you have any questions or issues:
1. Check the [Issues](https://github.com/Samuux23/CI_project/issues) page
2. Create a new issue with detailed information
3. Contact the maintainer

---

**🎉 Enjoy predicting personalities with AI!**

**[🚀 Try the Live App Now](https://personality-predictor.streamlit.app)** 
# Personality Classification Model

## Project Overview

This project implements a machine learning pipeline to classify individuals as either "Extrovert" or "Introvert" based on their behavioral and social characteristics. The model achieves **96.8% cross-validation accuracy** using a Gradient Boosting Classifier.

## Dataset

- **Training samples**: 18,524
- **Test samples**: 6,175
- **Features**: 7 (after preprocessing)
- **Target**: Personality (Extrovert/Introvert)
- **Class distribution**: 74% Extrovert, 26% Introvert

### Features

**Numerical Features:**
- `Time_spent_Alone`: Hours spent alone per day
- `Social_event_attendance`: Frequency of social event attendance
- `Going_outside`: Frequency of going outside
- `Friends_circle_size`: Number of friends
- `Post_frequency`: Frequency of social media posts

**Categorical Features:**
- `Stage_fear`: Whether the person experiences stage fear (Yes/No)
- `Drained_after_socializing`: Whether the person feels drained after socializing (Yes/No)

## Model Performance

### Best Model: Gradient Boosting Classifier
- **Cross-validation accuracy**: 96.8% (±0.62%)
- **Validation accuracy**: 97.1%
- **F1-score**: 94.5%

### Model Comparison
| Model | CV Accuracy | Validation Accuracy | F1-Score |
|-------|-------------|-------------------|----------|
| Gradient Boosting | 96.8% | 97.1% | 94.5% |
| Logistic Regression | 96.8% | 97.2% | 94.6% |
| LightGBM | 96.8% | 97.1% | 94.5% |
| CatBoost | 96.7% | 97.1% | 94.5% |
| XGBoost | 96.6% | 97.0% | 94.3% |
| Random Forest | 96.2% | 96.7% | 93.8% |

## Feature Importance

The most important features for personality classification are:

1. **Drained_after_socializing_Yes** (72.8%) - Most predictive feature
2. **Stage_fear_Yes** (14.7%) - Second most important
3. **Time_spent_Alone** (7.4%) - Moderate importance
4. **Going_outside** (2.5%) - Lower importance
5. **Social_event_attendance** (1.6%) - Lower importance
6. **Post_frequency** (0.5%) - Minimal importance
7. **Friends_circle_size** (0.4%) - Minimal importance

## Key Insights

1. **Social Energy**: Whether someone feels drained after socializing is the strongest predictor of introversion/extroversion
2. **Social Anxiety**: Stage fear is a significant indicator of introversion
3. **Solitude Preference**: Time spent alone is moderately predictive
4. **Social Activity**: Features like going outside and social event attendance have lower predictive power
5. **Digital Behavior**: Social media posting frequency and friend count are least predictive

## Files Generated

### Model Files
- `best_personality_model.pkl` - Trained Gradient Boosting model
- `preprocessor.pkl` - Data preprocessing pipeline
- `feature_names.pkl` - Feature names after preprocessing
- `label_encoder.pkl` - Label encoder for target variable

### Predictions
- `submission.csv` - Predictions for test set

### Visualizations
- `eda_visualizations.png` - Exploratory data analysis plots
- `confusion_matrix.png` - Model confusion matrix
- `feature_importance.png` - Feature importance plot

### Documentation
- `model_report.md` - Detailed model performance report
- `README.md` - This file

## Usage

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the complete pipeline:**
   ```bash
   python personality_classifier.py
   ```

3. **Test the model with examples:**
   ```bash
   python predict_example.py
   ```

### Using the Model in Your Code

```python
import joblib
import pandas as pd

# Load the model and components
model = joblib.load('best_personality_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Prepare new data (must have same features as training data)
new_data = pd.DataFrame({
    'Time_spent_Alone': [3.0],
    'Stage_fear': ['No'],
    'Social_event_attendance': [5.0],
    'Going_outside': [4.0],
    'Drained_after_socializing': ['No'],
    'Friends_circle_size': [8.0],
    'Post_frequency': [6.0]
})

# Make prediction
X_processed = preprocessor.transform(new_data)
prediction_encoded = model.predict(X_processed)
prediction = label_encoder.inverse_transform(prediction_encoded)

print(f"Predicted Personality: {prediction[0]}")
```

## Data Preprocessing

The pipeline handles:
- **Missing values**: Median imputation for numerical features, mode imputation for categorical
- **Categorical encoding**: One-hot encoding with first category dropped
- **Feature scaling**: StandardScaler for numerical features
- **Label encoding**: Target variable encoded as 0 (Introvert) and 1 (Extrovert)

## Model Selection Process

1. **Benchmarking**: Tested 6 different algorithms
2. **Cross-validation**: 5-fold stratified cross-validation
3. **Hyperparameter tuning**: Grid search for best models (when applicable)
4. **Feature importance analysis**: SHAP and permutation importance
5. **Final selection**: Gradient Boosting based on highest CV accuracy

## Technical Details

- **Cross-validation**: Stratified 5-fold CV to handle class imbalance
- **Evaluation metrics**: Accuracy, F1-score, precision, recall
- **Feature engineering**: Automatic feature type detection and appropriate preprocessing
- **Model persistence**: All components saved for easy deployment

## Performance Analysis

The model shows excellent performance with:
- High accuracy across all folds (96.8% ± 0.62%)
- Good balance between precision and recall
- Strong performance on both classes despite class imbalance
- Robust feature importance ranking

## Future Improvements

Potential enhancements:
1. **Ensemble methods**: Combine multiple models for better performance
2. **Feature engineering**: Create interaction features
3. **Advanced sampling**: SMOTE or other techniques for class imbalance
4. **Deep learning**: Neural networks for complex pattern recognition
5. **Hyperparameter optimization**: Bayesian optimization for better tuning

## Requirements

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.1.0
- xgboost >= 1.6.0
- lightgbm >= 3.3.0
- catboost >= 1.1.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- shap >= 0.41.0
- joblib >= 1.1.0

## License

This project is for educational and research purposes. 
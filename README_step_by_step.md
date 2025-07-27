# Step-by-Step Machine Learning Pipeline

This repository contains a comprehensive, step-by-step machine learning pipeline for personality classification (Extrovert/Introvert prediction).

## 🚀 Quick Start

### Option 1: Run the complete pipeline
```bash
python run_pipeline.py
```

### Option 2: Run step by step
```bash
python step_by_step_ml_pipeline.py
```

### Option 3: Google Colab
1. Upload the script to Google Colab
2. Upload your dataset files (train.csv, test.csv)
3. Run the script

## 📋 Pipeline Steps

The pipeline consists of 12 main steps:

### Step 1: Install Dependencies
- Automatically installs required Python packages
- Checks if packages are already installed

### Step 2: Import Libraries
- Imports all necessary ML libraries
- Sets up warnings and configurations

### Step 3: Load and Explore Data
- Loads training and test datasets
- Performs initial data exploration
- Checks for missing values and data types
- Analyzes target variable distribution

### Step 4: Create Preprocessing Pipeline
- Identifies numerical and categorical features
- Creates imputation strategies
- Sets up scaling and encoding pipelines

### Step 5: Model Definitions
- Defines 14 different model types
- **FIXED**: Removed `random_state` parameter from KNeighborsClassifier and GaussianNB
- Includes hyperparameter search spaces for each model

### Step 6: Optimization Objective
- Creates Optuna objective function
- Implements cross-validation with SMOTE
- Handles errors gracefully

### Step 7: Main Optimization Pipeline
- Runs hyperparameter optimization
- Uses 50 trials with TPE sampler
- Performs stratified k-fold cross-validation

### Step 8: Train Best Model
- Trains the best model with optimal parameters
- Creates complete pipeline with preprocessing and SMOTE

### Step 9: Evaluate Model
- Evaluates on validation set
- Generates classification report and confusion matrix
- Calculates accuracy metrics

### Step 10: Generate Predictions
- Creates predictions for test set
- Saves submission file in correct format
- Shows prediction distribution

### Step 11: Save Model and Results
- Saves trained model as joblib file
- Saves optimization study
- Creates configuration JSON file

### Step 12: Create Summary Report
- Generates comprehensive markdown report
- Includes all results and configurations

## 📁 Files Generated

After running the pipeline, you'll get:

- `best_model_step_by_step.joblib` - Trained model
- `optimization_study_step_by_step.pkl` - Optimization results
- `best_config_step_by_step.json` - Best configuration
- `submission_step_by_step.csv` - Test predictions
- `summary_report_step_by_step.md` - Summary report

## 🔧 Models Included

1. **LogisticRegression** - Linear classification
2. **RandomForest** - Ensemble tree method
3. **XGBoost** - Gradient boosting
4. **LightGBM** - Light gradient boosting
5. **CatBoost** - Categorical boosting
6. **SVC** - Support Vector Classification
7. **KNeighbors** - K-Nearest Neighbors
8. **GaussianNB** - Naive Bayes
9. **MLPClassifier** - Neural Network
10. **GradientBoosting** - Gradient Boosting
11. **ExtraTrees** - Extra Trees
12. **DecisionTree** - Decision Tree
13. **RidgeClassifier** - Ridge Classification
14. **SGDClassifier** - Stochastic Gradient Descent

## 🛠️ Installation

### Local Installation
```bash
pip install -r requirements_step_by_step.txt
```

### Google Colab
The script automatically installs dependencies when run in Colab.

## 📊 Expected Results

Based on previous runs, you can expect:
- **Best Model**: Usually GradientBoosting or CatBoost
- **Best Accuracy**: ~96.9%
- **Optimization Time**: 5-10 minutes
- **Total Trials**: 50

## 🐛 Error Fixes

This version includes fixes for common errors:
- ✅ Removed `random_state` from KNeighborsClassifier
- ✅ Removed `random_state` from GaussianNB
- ✅ Proper error handling in optimization
- ✅ Correct parameter handling for all models

## 📈 Performance Monitoring

The pipeline includes:
- Progress bars for optimization
- Detailed logging of each step
- Error handling and recovery
- Performance metrics and reports

## 🎯 Usage Examples

### Basic Usage
```python
# Run complete pipeline
python run_pipeline.py
```

### Custom Optimization
```python
# Modify the number of trials in step_by_step_ml_pipeline.py
N_TRIALS = 100  # Change from 50 to 100 for more thorough search
```

### Load Saved Model
```python
import joblib

# Load the trained model
model = joblib.load('best_model_step_by_step.joblib')

# Make predictions
predictions = model.predict(new_data)
```

## 📝 Notes

- The pipeline automatically handles missing values
- SMOTE is used to handle class imbalance
- Cross-validation ensures robust evaluation
- All results are saved for reproducibility
- The script works both locally and in Google Colab

## 🤝 Support

If you encounter any issues:
1. Check that all required files are present
2. Ensure you have sufficient memory for optimization
3. Verify Python version compatibility (3.7+)
4. Check the error logs in the console output 
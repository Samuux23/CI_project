"""
STEP-BY-STEP MACHINE LEARNING PIPELINE
Personality Classification (Extrovert/Introvert)

This script provides a complete, step-by-step approach to:
1. Data loading and exploration
2. Data preprocessing
3. Model training and optimization
4. Evaluation and prediction
5. Saving results

Author: ML Engineer
Date: 2024
"""

# ============================================================================
# STEP 1: INSTALL DEPENDENCIES
# ============================================================================

import subprocess
import sys

def install_required_packages():
    """Install required packages if not already installed"""
    print("=== STEP 1: INSTALLING DEPENDENCIES ===")
    
    packages = [
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'scikit-learn>=1.1.0',
        'xgboost>=1.6.0',
        'lightgbm>=3.3.0',
        'catboost>=1.1.0',
        'optuna>=3.0.0',
        'imbalanced-learn>=0.9.0',
        'joblib>=1.1.0'
    ]
    
    for package in packages:
        try:
            __import__(package.split('>=')[0])
            print(f"✓ {package.split('>=')[0]} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ {package} installed successfully")

# ============================================================================
# STEP 2: IMPORT LIBRARIES
# ============================================================================

def import_libraries():
    """Import all required libraries"""
    print("\n=== STEP 2: IMPORTING LIBRARIES ===")
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    import optuna
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    import joblib
    import warnings
    import json
    from datetime import datetime
    
    warnings.filterwarnings('ignore')
    
    print("✓ All libraries imported successfully")
    return locals()

# ============================================================================
# STEP 3: LOAD AND EXPLORE DATA
# ============================================================================

def load_and_explore_data():
    """Load data and perform initial exploration"""
    print("\n=== STEP 3: LOADING AND EXPLORING DATA ===")
    
    # Load data
    print("Loading training data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Display basic info
    print("\nTraining data info:")
    print(train_df.info())
    
    print("\nFirst few rows of training data:")
    print(train_df.head())
    
    # Check for missing values
    print("\nMissing values in training data:")
    print(train_df.isnull().sum())
    
    # Check target distribution
    print("\nTarget variable distribution:")
    print(train_df['Personality'].value_counts())
    print(train_df['Personality'].value_counts(normalize=True))
    
    # Basic statistics
    print("\nNumerical features statistics:")
    numerical_cols = train_df.select_dtypes(include=[np.number]).columns
    print(train_df[numerical_cols].describe())
    
    return train_df, test_df

# ============================================================================
# STEP 4: DATA PREPROCESSING
# ============================================================================

def create_preprocessing_pipeline():
    """Create preprocessing pipeline for numerical and categorical features"""
    print("\n=== STEP 4: CREATING PREPROCESSING PIPELINE ===")
    
    # Identify feature types
    train_df = pd.read_csv('train.csv')
    X_sample = train_df.drop(['id', 'Personality'], axis=1)
    
    numerical_features = X_sample.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_sample.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Numerical transformer
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    print("✓ Preprocessing pipeline created successfully")
    return preprocessor

# ============================================================================
# STEP 5: MODEL DEFINITIONS (FIXED VERSIONS)
# ============================================================================

def get_model_and_params(trial):
    """Get model and parameters based on trial suggestion (FIXED VERSION)"""
    model_name = trial.suggest_categorical('model', [
        'LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM', 
        'CatBoost', 'SVC', 'KNeighbors', 'GaussianNB', 'MLPClassifier',
        'GradientBoosting', 'ExtraTrees', 'DecisionTree', 'RidgeClassifier', 'SGDClassifier'
    ])
    
    if model_name == 'LogisticRegression':
        params = {
            'C': trial.suggest_float('C', 0.001, 100, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
            'max_iter': trial.suggest_int('max_iter', 100, 1000)
        }
        model = LogisticRegression(**params, random_state=42)
        
    elif model_name == 'RandomForest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }
        model = RandomForestClassifier(**params, random_state=42)
        
    elif model_name == 'XGBoost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
        }
        model = xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss')
        
    elif model_name == 'LightGBM':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }
        model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
        
    elif model_name == 'CatBoost':
        params = {
            'iterations': trial.suggest_int('iterations', 50, 500),
            'depth': trial.suggest_int('depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10)
        }
        model = cb.CatBoostClassifier(**params, random_state=42, verbose=False)
        
    elif model_name == 'SVC':
        params = {
            'C': trial.suggest_float('C', 0.1, 100),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
        }
        model = SVC(**params, random_state=42)
        
    elif model_name == 'KNeighbors':
        # FIXED: Removed random_state parameter
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'p': trial.suggest_int('p', 1, 3)
        }
        model = KNeighborsClassifier(**params)
        
    elif model_name == 'GaussianNB':
        # FIXED: Removed random_state parameter
        params = {
            'var_smoothing': trial.suggest_float('var_smoothing', 1e-9, 1e-6)
        }
        model = GaussianNB(**params)
        
    elif model_name == 'MLPClassifier':
        params = {
            'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', 
                [(50,), (100,), (50, 50), (100, 50)]),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'alpha': trial.suggest_float('alpha', 0.0001, 1),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 0.001, 0.1),
            'max_iter': trial.suggest_int('max_iter', 200, 500)
        }
        model = MLPClassifier(**params, random_state=42)
        
    elif model_name == 'GradientBoosting':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }
        model = GradientBoostingClassifier(**params, random_state=42)
        
    elif model_name == 'ExtraTrees':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }
        model = ExtraTreesClassifier(**params, random_state=42)
        
    elif model_name == 'DecisionTree':
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }
        model = DecisionTreeClassifier(**params, random_state=42)
        
    elif model_name == 'RidgeClassifier':
        params = {
            'alpha': trial.suggest_float('alpha', 0.1, 10),
            'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
        }
        model = RidgeClassifier(**params, random_state=42)
        
    elif model_name == 'SGDClassifier':
        params = {
            'loss': trial.suggest_categorical('loss', ['hinge', 'log', 'modified_huber']),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
            'alpha': trial.suggest_float('alpha', 0.0001, 1),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive'])
        }
        model = SGDClassifier(**params, random_state=42)
    
    return model, model_name

# ============================================================================
# STEP 6: OPTIMIZATION OBJECTIVE
# ============================================================================

def objective(trial, X_train, y_train):
    """Optuna objective function for hyperparameter optimization"""
    try:
        # Get model and parameters
        model, model_name = get_model_and_params(trial)
        
        # Create preprocessing pipeline
        preprocessor = create_preprocessing_pipeline()
        
        # Create full pipeline with SMOTE
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
        
        return scores.mean()
    except Exception as e:
        print(f"Error in trial: {e}")
        return 0.0

# ============================================================================
# STEP 7: MAIN OPTIMIZATION PIPELINE
# ============================================================================

def run_optimization():
    """Run the complete optimization pipeline"""
    print("\n=== STEP 7: RUNNING OPTIMIZATION PIPELINE ===")
    
    # Load data
    train_df, test_df = load_and_explore_data()
    
    # Prepare features and target
    X = train_df.drop(['id', 'Personality'], axis=1)
    y = train_df['Personality']
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Run optimization
    print("Starting optimization with 50 trials...")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50, show_progress_bar=True)
    
    # Print best results
    print(f"\nBest trial:")
    print(f"  Value: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")
    
    return study, X_train, X_val, y_train, y_val, test_df

# ============================================================================
# STEP 8: TRAIN BEST MODEL
# ============================================================================

def train_best_model(study, X_train, y_train):
    """Train the best model with optimal parameters"""
    print("\n=== STEP 8: TRAINING BEST MODEL ===")
    
    # Get best model
    best_model, best_model_name = get_model_and_params(optuna.trial.FixedTrial(study.best_params))
    preprocessor = create_preprocessing_pipeline()
    
    # Create best pipeline
    best_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', best_model)
    ])
    
    # Train best model
    print(f"Training {best_model_name} with best parameters...")
    best_pipeline.fit(X_train, y_train)
    
    print("✓ Best model trained successfully!")
    return best_pipeline, best_model_name

# ============================================================================
# STEP 9: EVALUATE MODEL
# ============================================================================

def evaluate_model(best_pipeline, X_val, y_val):
    """Evaluate the best model on validation set"""
    print("\n=== STEP 9: EVALUATING MODEL ===")
    
    # Make predictions
    y_pred = best_pipeline.predict(X_val)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    
    return accuracy

# ============================================================================
# STEP 10: GENERATE PREDICTIONS
# ============================================================================

def generate_predictions(best_pipeline, test_df):
    """Generate predictions for test set"""
    print("\n=== STEP 10: GENERATING PREDICTIONS ===")
    
    # Prepare test features
    X_test = test_df.drop(['id'], axis=1)
    
    # Make predictions
    predictions = best_pipeline.predict(X_test)
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': predictions
    })
    
    # Save predictions
    submission.to_csv('submission_step_by_step.csv', index=False)
    
    print("✓ Predictions saved to 'submission_step_by_step.csv'")
    
    # Show prediction distribution
    print("\nPrediction distribution:")
    print(submission['Personality'].value_counts())
    print(submission['Personality'].value_counts(normalize=True))
    
    return submission

# ============================================================================
# STEP 11: SAVE MODEL AND RESULTS
# ============================================================================

def save_model_and_results(best_pipeline, study, best_model_name, accuracy):
    """Save the best model and optimization results"""
    print("\n=== STEP 11: SAVING MODEL AND RESULTS ===")
    
    # Save best model
    joblib.dump(best_pipeline, 'best_model_step_by_step.joblib')
    print("✓ Best model saved to 'best_model_step_by_step.joblib'")
    
    # Save study
    joblib.dump(study, 'optimization_study_step_by_step.pkl')
    print("✓ Optimization study saved to 'optimization_study_step_by_step.pkl'")
    
    # Save best configuration
    best_config = {
        'model_name': best_model_name,
        'best_score': study.best_value,
        'best_params': study.best_params,
        'validation_accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('best_config_step_by_step.json', 'w') as f:
        json.dump(best_config, f, indent=2)
    print("✓ Best configuration saved to 'best_config_step_by_step.json'")
    
    return best_config

# ============================================================================
# STEP 12: CREATE SUMMARY REPORT
# ============================================================================

def create_summary_report(best_config, study):
    """Create a summary report of the optimization results"""
    print("\n=== STEP 12: CREATING SUMMARY REPORT ===")
    
    report = f"""
# MACHINE LEARNING PIPELINE SUMMARY REPORT

## Optimization Results
- **Best Model**: {best_config['model_name']}
- **Best Cross-Validation Score**: {best_config['best_score']:.4f}
- **Validation Accuracy**: {best_config['validation_accuracy']:.4f}
- **Total Trials**: {len(study.trials)}
- **Optimization Date**: {best_config['timestamp']}

## Best Hyperparameters
```json
{json.dumps(best_config['best_params'], indent=2)}
```

## Model Performance
The best model achieved a cross-validation accuracy of {best_config['best_score']:.4f} 
and validation accuracy of {best_config['validation_accuracy']:.4f}.

## Files Generated
- `best_model_step_by_step.joblib`: Trained model
- `optimization_study_step_by_step.pkl`: Optimization study
- `best_config_step_by_step.json`: Best configuration
- `submission_step_by_step.csv`: Test predictions

## Next Steps
1. Use the saved model for new predictions
2. Monitor model performance in production
3. Retrain periodically with new data
"""
    
    with open('summary_report_step_by_step.md', 'w') as f:
        f.write(report)
    
    print("✓ Summary report saved to 'summary_report_step_by_step.md'")
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main execution function - runs all steps"""
    print("🚀 STARTING STEP-BY-STEP MACHINE LEARNING PIPELINE")
    print("="*60)
    
    try:
        # Step 1: Install dependencies
        install_required_packages()
        
        # Step 2: Import libraries
        globals().update(import_libraries())
        
        # Step 3-7: Run optimization
        study, X_train, X_val, y_train, y_val, test_df = run_optimization()
        
        # Step 8: Train best model
        best_pipeline, best_model_name = train_best_model(study, X_train, y_train)
        
        # Step 9: Evaluate model
        accuracy = evaluate_model(best_pipeline, X_val, y_val)
        
        # Step 10: Generate predictions
        submission = generate_predictions(best_pipeline, test_df)
        
        # Step 11: Save model and results
        best_config = save_model_and_results(best_pipeline, study, best_model_name, accuracy)
        
        # Step 12: Create summary report
        create_summary_report(best_config, study)
        
        print(f"\n🎉 PIPELINE COMPLETED! Best model: {best_model_name}")
        print(f"📊 Best accuracy: {study.best_value:.4f}")
        print(f"📁 Results saved in current directory")
        
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import warnings
import joblib
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import time
from datetime import datetime
import json

warnings.filterwarnings('ignore')

class ComprehensiveHyperparameterOptimizer:
    def __init__(self, n_trials=30, cv_folds=5, random_state=42):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.best_score = 0
        self.best_model_name = None
        self.all_results = []
        
        # Set random seeds
        np.random.seed(random_state)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
    def load_and_analyze_data(self):
        """Load and analyze the dataset"""
        print("=== DATA LOADING AND ANALYSIS ===")
        
        # Load data
        self.train_data = pd.read_csv('train.csv')
        self.test_data = pd.read_csv('test.csv')
        
        print(f"Training data shape: {self.train_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
        
        # Separate features and target
        X = self.train_data.drop(['id', 'Personality'], axis=1)
        y = self.train_data['Personality']
        
        # Encode target
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Feature analysis
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"\nFeature Analysis:")
        print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
        print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
        
        # Missing value analysis
        print(f"\nMissing Values Analysis:")
        missing_counts = X.isnull().sum()
        for col, count in missing_counts[missing_counts > 0].items():
            percentage = count/len(X)*100
            print(f"  {col}: {count} ({percentage:.1f}%)")
        
        # Class balance analysis
        class_counts = y.value_counts()
        print(f"\nClass Distribution:")
        for class_name, count in class_counts.items():
            percentage = count/len(y)*100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Store data
        self.X_raw = X
        self.y_raw = y
        self.y_encoded = y_encoded
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        
        return X, y_encoded
    
    def create_preprocessing_pipeline(self):
        """Create comprehensive preprocessing pipeline"""
        print("\n=== CREATING PREPROCESSING PIPELINE ===")
        
        # Numerical features: Try multiple imputation strategies
        numerical_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),  # More sophisticated than median
            ('scaler', RobustScaler())  # More robust than StandardScaler
        ])
        
        # Categorical features: One-hot encoding
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        # Apply preprocessing
        X_processed = self.preprocessor.fit_transform(self.X_raw)
        
        # Get feature names
        numerical_feature_names = self.numerical_features
        categorical_feature_names = []
        for i, feature in enumerate(self.categorical_features):
            categories = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[i][1:]
            categorical_feature_names.extend([f"{feature}_{cat}" for cat in categories])
        
        self.feature_names = numerical_feature_names + categorical_feature_names
        
        print(f"Processed features shape: {X_processed.shape}")
        print(f"Number of features after preprocessing: {len(self.feature_names)}")
        
        self.X = X_processed
        self.y = self.y_encoded
        
        return X_processed, self.y_encoded
    
    def define_model_configurations(self):
        """Define all model families and their hyperparameter spaces"""
        self.model_configs = {
            # Linear Models
            'LogisticRegression': {
                'model_class': LogisticRegression,
                'params': {
                    'C': (0.001, 100),
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga'],
                    'max_iter': (100, 2000)
                }
            },
            'RidgeClassifier': {
                'model_class': RidgeClassifier,
                'params': {
                    'alpha': (0.001, 100),
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
                }
            },
            'SGDClassifier': {
                'model_class': SGDClassifier,
                'params': {
                    'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge'],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'alpha': (0.0001, 1),
                    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                    'eta0': (0.01, 1)
                }
            },
            
            # Tree-based Models
            'DecisionTree': {
                'model_class': DecisionTreeClassifier,
                'params': {
                    'max_depth': (3, 20),
                    'min_samples_split': (2, 20),
                    'min_samples_leaf': (1, 10),
                    'max_features': ['sqrt', 'log2', None],
                    'criterion': ['gini', 'entropy']
                }
            },
            'RandomForest': {
                'model_class': RandomForestClassifier,
                'params': {
                    'n_estimators': (50, 500),
                    'max_depth': (3, 20),
                    'min_samples_split': (2, 20),
                    'min_samples_leaf': (1, 10),
                    'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True, False]
                }
            },
            'ExtraTrees': {
                'model_class': ExtraTreesClassifier,
                'params': {
                    'n_estimators': (50, 500),
                    'max_depth': (3, 20),
                    'min_samples_split': (2, 20),
                    'min_samples_leaf': (1, 10),
                    'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True, False]
                }
            },
            'GradientBoosting': {
                'model_class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': (50, 500),
                    'max_depth': (3, 15),
                    'learning_rate': (0.01, 0.3),
                    'subsample': (0.6, 1.0),
                    'min_samples_split': (2, 20),
                    'min_samples_leaf': (1, 10),
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            
            # Boosting Models
            'XGBoost': {
                'model_class': xgb.XGBClassifier,
                'params': {
                    'n_estimators': (50, 500),
                    'max_depth': (3, 15),
                    'learning_rate': (0.01, 0.3),
                    'subsample': (0.6, 1.0),
                    'colsample_bytree': (0.6, 1.0),
                    'reg_alpha': (0, 10),
                    'reg_lambda': (0, 10),
                    'min_child_weight': (1, 10)
                }
            },
            'LightGBM': {
                'model_class': lgb.LGBMClassifier,
                'params': {
                    'n_estimators': (50, 500),
                    'max_depth': (3, 15),
                    'learning_rate': (0.01, 0.3),
                    'num_leaves': (20, 300),
                    'subsample': (0.6, 1.0),
                    'colsample_bytree': (0.6, 1.0),
                    'reg_alpha': (0, 10),
                    'reg_lambda': (0, 10),
                    'min_child_samples': (10, 100)
                }
            },
            'CatBoost': {
                'model_class': cb.CatBoostClassifier,
                'params': {
                    'iterations': (50, 500),
                    'depth': (3, 15),
                    'learning_rate': (0.01, 0.3),
                    'l2_leaf_reg': (1, 10),
                    'border_count': (32, 255),
                    'bagging_temperature': (0, 1),
                    'random_strength': (0, 10)
                }
            },
            
            # Support Vector Machines
            'SVC': {
                'model_class': SVC,
                'params': {
                    'C': (0.1, 100),
                    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto'],
                    'degree': (2, 5)  # Only for poly kernel
                }
            },
            
            # Nearest Neighbors
            'KNeighbors': {
                'model_class': KNeighborsClassifier,
                'params': {
                    'n_neighbors': (3, 20),
                    'weights': ['uniform', 'distance'],
                    'p': (1, 3),  # Manhattan vs Euclidean
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }
            },
            
            # Naive Bayes
            'GaussianNB': {
                'model_class': GaussianNB,
                'params': {
                    'var_smoothing': (1e-9, 1e-6)
                }
            },
            
            # Neural Networks
            'MLPClassifier': {
                'model_class': MLPClassifier,
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                    'activation': ['relu', 'tanh', 'logistic'],
                    'alpha': (0.0001, 1),
                    'learning_rate_init': (0.001, 0.1),
                    'max_iter': (200, 1000)
                }
            }
        }
        
        print(f"Defined {len(self.model_configs)} model families:")
        for model_name in self.model_configs.keys():
            print(f"  - {model_name}")
        
        return self.model_configs
    
    def sample_hyperparameters(self, trial, model_name):
        """Sample hyperparameters for a given model"""
        config = self.model_configs[model_name]
        params = {}
        
        for param_name, distribution in config['params'].items():
            if isinstance(distribution, list):
                # Categorical parameter
                params[param_name] = trial.suggest_categorical(param_name, distribution)
            elif isinstance(distribution, tuple):
                # Numerical parameter
                if isinstance(distribution[0], int):
                    params[param_name] = trial.suggest_int(param_name, distribution[0], distribution[1])
                else:
                    params[param_name] = trial.suggest_float(param_name, distribution[0], distribution[1])
        
        # Add random_state for reproducibility
        if 'random_state' not in params:
            params['random_state'] = self.random_state
        
        return params
    
    def create_model(self, model_name, params):
        """Create model instance with given parameters"""
        config = self.model_configs[model_name]
        model_class = config['model_class']
        
        # Handle special cases
        if model_name == 'SVC' and params.get('kernel') == 'poly':
            # Ensure degree is set for poly kernel
            if 'degree' not in params:
                params['degree'] = 3
        elif model_name == 'SVC' and params.get('kernel') != 'poly':
            # Remove degree if not poly kernel
            params.pop('degree', None)
        
        # Create model
        try:
            model = model_class(**params)
            return model
        except Exception as e:
            print(f"Error creating {model_name} with params {params}: {e}")
            return None
    
    def objective(self, trial, model_name):
        """Objective function for Optuna optimization"""
        # Sample hyperparameters
        params = self.sample_hyperparameters(trial, model_name)
        
        # Create model
        model = self.create_model(model_name, params)
        if model is None:
            return 0.0
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        try:
            # For models that support early stopping
            if model_name in ['XGBoost', 'LightGBM', 'CatBoost']:
                scores = []
                for train_idx, val_idx in cv.split(self.X, self.y):
                    X_train, X_val = self.X[train_idx], self.X[val_idx]
                    y_train, y_val = self.y[train_idx], self.y[val_idx]
                    
                    if model_name == 'XGBoost':
                        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
                    elif model_name == 'LightGBM':
                        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                    elif model_name == 'CatBoost':
                        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
                    
                    y_pred = model.predict(X_val)
                    score = accuracy_score(y_val, y_pred)
                    scores.append(score)
            else:
                scores = cross_val_score(model, self.X, self.y, cv=cv, scoring='accuracy', n_jobs=-1)
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Store results
            result = {
                'trial': trial.number,
                'model': model_name,
                'params': params.copy(),
                'mean_score': mean_score,
                'std_score': std_score,
                'scores': scores
            }
            self.all_results.append(result)
            
            return mean_score
            
        except Exception as e:
            print(f"Error in trial {trial.number} for {model_name}: {e}")
            return 0.0
    
    def optimize_model(self, model_name):
        """Optimize hyperparameters for a specific model"""
        print(f"\n=== OPTIMIZING {model_name.upper()} ===")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        
        # Optimize
        start_time = time.time()
        study.optimize(
            lambda trial: self.objective(trial, model_name),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        end_time = time.time()
        
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        print(f"Best score: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")
        
        return study
    
    def optimize_all_models(self):
        """Optimize all models and find the best one"""
        print("\n=== COMPREHENSIVE HYPERPARAMETER OPTIMIZATION ===")
        
        studies = {}
        
        for model_name in self.model_configs.keys():
            try:
                study = self.optimize_model(model_name)
                studies[model_name] = study
                
                # Update best model if this is better
                if study.best_value > self.best_score:
                    self.best_score = study.best_value
                    self.best_params = study.best_params
                    self.best_model_name = model_name
                    self.best_study = study
                    
            except Exception as e:
                print(f"Failed to optimize {model_name}: {e}")
                continue
        
        self.studies = studies
        return studies
    
    def create_best_model(self):
        """Create and train the best model with optimal parameters"""
        print(f"\n=== CREATING BEST MODEL: {self.best_model_name} ===")
        
        # Create best model
        self.best_model = self.create_model(self.best_model_name, self.best_params)
        
        # Train on full dataset
        self.best_model.fit(self.X, self.y)
        
        print(f"Best model trained successfully!")
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {self.best_score:.4f}")
        
        return self.best_model
    
    def generate_comprehensive_report(self):
        """Generate comprehensive results report"""
        print("\n=== GENERATING COMPREHENSIVE REPORT ===")
        
        # Create results DataFrame
        results_df = []
        for result in self.all_results:
            results_df.append({
                'Model': result['model'],
                'Trial': result['trial'],
                'Mean_Score': result['mean_score'],
                'Std_Score': result['std_score'],
                'Best_Params': str(result['params'])
            })
        
        results_df = pd.DataFrame(results_df)
        
        # Get best results for each model
        best_results = results_df.loc[results_df.groupby('Model')['Mean_Score'].idxmax()]
        best_results = best_results.sort_values('Mean_Score', ascending=False)
        
        print("\n=== TOP PERFORMING CONFIGURATIONS ===")
        print(best_results[['Model', 'Mean_Score', 'Std_Score']].to_string(index=False))
        
        # Save detailed results
        results_df.to_csv('comprehensive_hyperparameter_results.csv', index=False)
        best_results.to_csv('best_configurations_comprehensive.csv', index=False)
        
        # Save best parameters as JSON
        with open('best_model_config.json', 'w') as f:
            json.dump({
                'model_name': self.best_model_name,
                'best_score': self.best_score,
                'best_params': self.best_params
            }, f, indent=2)
        
        return results_df, best_results
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n=== CREATING COMPREHENSIVE VISUALIZATIONS ===")
        
        # 1. Optimization history for each model
        n_models = len(self.studies)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        fig.suptitle('Hyperparameter Optimization History - All Models', fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_name, study) in enumerate(self.studies.items()):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            trials_df = study.trials_dataframe()
            ax.plot(trials_df['number'], trials_df['value'], 'b-', alpha=0.6)
            ax.set_title(f'{model_name}\nBest: {study.best_value:.4f}')
            ax.set_xlabel('Trial')
            ax.set_ylabel('Accuracy')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_models, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('comprehensive_optimization_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Best scores comparison
        plt.figure(figsize=(14, 8))
        model_names = list(self.studies.keys())
        best_scores = [study.best_value for study in self.studies.values()]
        
        bars = plt.bar(model_names, best_scores, color='skyblue', alpha=0.7)
        plt.title('Best Cross-Validation Scores - All Model Families', fontsize=14, fontweight='bold')
        plt.xlabel('Model Family')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, best_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('comprehensive_best_scores_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Parameter importance for best model
        if hasattr(self.best_study, 'get_param_importances'):
            try:
                param_importance = self.best_study.get_param_importances()
                
                plt.figure(figsize=(10, 6))
                params = list(param_importance.keys())
                importances = list(param_importance.values())
                
                bars = plt.barh(params, importances, color='lightcoral', alpha=0.7)
                plt.title(f'Parameter Importance - {self.best_model_name}', fontsize=14, fontweight='bold')
                plt.xlabel('Importance')
                plt.ylabel('Parameter')
                plt.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, importance in zip(bars, importances):
                    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                            f'{importance:.3f}', ha='left', va='center', fontweight='bold')
                
                plt.tight_layout()
                plt.savefig('comprehensive_parameter_importance.png', dpi=300, bbox_inches='tight')
                plt.show()
            except:
                print("Parameter importance not available for this model")
        
        # 4. Feature importance for best model
        if hasattr(self.best_model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            importances = self.best_model.feature_importances_
            
            # Get top 15 features
            indices = np.argsort(importances)[::-1][:15]
            
            plt.bar(range(len(indices)), importances[indices], color='lightgreen', alpha=0.7)
            plt.title(f'Top 15 Feature Importances - {self.best_model_name}', fontsize=14, fontweight='bold')
            plt.xlabel('Feature Index')
            plt.ylabel('Importance')
            plt.xticks(range(len(indices)), [self.feature_names[i] for i in indices], rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('comprehensive_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_best_model(self):
        """Save the best model and all components"""
        print("\n=== SAVING BEST MODEL ===")
        
        # Save model
        joblib.dump(self.best_model, 'best_comprehensive_model.pkl')
        
        # Save preprocessor
        joblib.dump(self.preprocessor, 'preprocessor_comprehensive.pkl')
        
        # Save label encoder
        joblib.dump(self.label_encoder, 'label_encoder_comprehensive.pkl')
        
        # Save feature names
        joblib.dump(self.feature_names, 'feature_names_comprehensive.pkl')
        
        # Save best parameters
        joblib.dump(self.best_params, 'best_params_comprehensive.pkl')
        
        # Save optimization results
        joblib.dump(self.all_results, 'comprehensive_optimization_results.pkl')
        
        print("All components saved successfully!")
        print(f"Best model: {self.best_model_name}")
        print(f"Best score: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")
    
    def generate_predictions(self):
        """Generate predictions for test set"""
        print("\n=== GENERATING PREDICTIONS ===")
        
        # Load test data
        test_data = pd.read_csv('test.csv')
        X_test = test_data.drop(['id'], axis=1)
        
        # Preprocess test data
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Make predictions
        predictions_encoded = self.best_model.predict(X_test_processed)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        # Create submission
        submission = pd.DataFrame({
            'id': test_data['id'],
            'Personality': predictions
        })
        
        submission.to_csv('submission_comprehensive.csv', index=False)
        
        print("Predictions saved to 'submission_comprehensive.csv'")
        
        # Show prediction distribution
        pred_counts = submission['Personality'].value_counts()
        print(f"Prediction distribution:")
        for personality, count in pred_counts.items():
            print(f"  {personality}: {count} ({count/len(submission)*100:.1f}%)")
        
        return submission
    
    def run_complete_optimization(self):
        """Run the complete comprehensive hyperparameter optimization pipeline"""
        print("=== COMPREHENSIVE HYPERPARAMETER OPTIMIZATION PIPELINE ===\n")
        
        # Load and analyze data
        self.load_and_analyze_data()
        
        # Create preprocessing pipeline
        self.create_preprocessing_pipeline()
        
        # Define model configurations
        self.define_model_configurations()
        
        # Optimize all models
        self.optimize_all_models()
        
        # Create best model
        self.create_best_model()
        
        # Generate comprehensive report
        results_df, best_results = self.generate_comprehensive_report()
        
        # Create visualizations
        self.create_comprehensive_visualizations()
        
        # Save best model
        self.save_best_model()
        
        # Generate predictions
        submission = self.generate_predictions()
        
        print("\n=== COMPREHENSIVE OPTIMIZATION PIPELINE COMPLETED ===")
        print(f"Best model: {self.best_model_name}")
        print(f"Best cross-validation accuracy: {self.best_score:.4f}")
        print(f"Total trials completed: {len(self.all_results)}")
        print(f"Models tested: {len(self.studies)}")
        
        return self.best_model, self.best_score, self.best_params

if __name__ == "__main__":
    # Create optimizer
    optimizer = ComprehensiveHyperparameterOptimizer(n_trials=20, cv_folds=5, random_state=42)
    
    # Run complete optimization
    best_model, best_score, best_params = optimizer.run_complete_optimization() 
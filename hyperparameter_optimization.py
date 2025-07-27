import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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

warnings.filterwarnings('ignore')

class HyperparameterOptimizer:
    def __init__(self, n_trials=100, cv_folds=5, random_state=42):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.best_score = 0
        self.study = None
        self.results = []
        
        # Set random seeds
        np.random.seed(random_state)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        print("=== DATA LOADING AND PREPARATION ===")
        
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
        
        # Identify feature types
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"\nFeature Analysis:")
        print(f"Numerical features: {numerical_features}")
        print(f"Categorical features: {categorical_features}")
        
        # Missing value analysis
        print(f"\nMissing values:")
        missing_counts = X.isnull().sum()
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  {col}: {count} ({count/len(X)*100:.1f}%)")
        
        # Class balance
        class_counts = y.value_counts()
        print(f"\nClass distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} ({count/len(y)*100:.1f}%)")
        
        # Create preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Apply preprocessing
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names
        numerical_feature_names = numerical_features
        categorical_feature_names = []
        for i, feature in enumerate(categorical_features):
            categories = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[i][1:]
            categorical_feature_names.extend([f"{feature}_{cat}" for cat in categories])
        
        self.feature_names = numerical_feature_names + categorical_feature_names
        
        print(f"\nProcessed features shape: {X_processed.shape}")
        print(f"Number of features after preprocessing: {len(self.feature_names)}")
        
        self.X = X_processed
        self.y = y_encoded
        
        return X_processed, y_encoded
    
    def define_hyperparameter_space(self, model_name):
        """Define hyperparameter search space for different models"""
        if model_name == "RandomForest":
            return {
                'n_estimators': (50, 1000),
                'max_depth': (3, 20),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
        
        elif model_name == "XGBoost":
            return {
                'n_estimators': (50, 1000),
                'max_depth': (3, 15),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'reg_alpha': (0, 10),
                'reg_lambda': (0, 10),
                'min_child_weight': (1, 10)
            }
        
        elif model_name == "LightGBM":
            return {
                'n_estimators': (50, 1000),
                'max_depth': (3, 15),
                'learning_rate': (0.01, 0.3),
                'num_leaves': (20, 300),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'reg_alpha': (0, 10),
                'reg_lambda': (0, 10),
                'min_child_samples': (10, 100)
            }
        
        elif model_name == "CatBoost":
            return {
                'iterations': (50, 1000),
                'depth': (3, 15),
                'learning_rate': (0.01, 0.3),
                'l2_leaf_reg': (1, 10),
                'border_count': (32, 255),
                'bagging_temperature': (0, 1),
                'random_strength': (0, 10)
            }
        
        elif model_name == "GradientBoosting":
            return {
                'n_estimators': (50, 1000),
                'max_depth': (3, 15),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.6, 1.0),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'max_features': ['sqrt', 'log2', None]
            }
        
        elif model_name == "LogisticRegression":
            return {
                'C': (0.001, 100),
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': (100, 2000)
            }
    
    def create_model(self, model_name, params):
        """Create model with given parameters"""
        if model_name == "RandomForest":
            return RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                bootstrap=params['bootstrap'],
                random_state=self.random_state,
                n_jobs=-1
            )
        
        elif model_name == "XGBoost":
            return xgb.XGBClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                reg_alpha=params['reg_alpha'],
                reg_lambda=params['reg_lambda'],
                min_child_weight=params['min_child_weight'],
                random_state=self.random_state,
                eval_metric='logloss',
                verbose=0
            )
        
        elif model_name == "LightGBM":
            return lgb.LGBMClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                num_leaves=params['num_leaves'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                reg_alpha=params['reg_alpha'],
                reg_lambda=params['reg_lambda'],
                min_child_samples=params['min_child_samples'],
                random_state=self.random_state,
                verbose=-1
            )
        
        elif model_name == "CatBoost":
            return cb.CatBoostClassifier(
                iterations=params['iterations'],
                depth=params['depth'],
                learning_rate=params['learning_rate'],
                l2_leaf_reg=params['l2_leaf_reg'],
                border_count=params['border_count'],
                bagging_temperature=params['bagging_temperature'],
                random_strength=params['random_strength'],
                random_state=self.random_state,
                verbose=False
            )
        
        elif model_name == "GradientBoosting":
            return GradientBoostingClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                random_state=self.random_state
            )
        
        elif model_name == "LogisticRegression":
            return LogisticRegression(
                C=params['C'],
                penalty=params['penalty'],
                solver=params['solver'],
                max_iter=params['max_iter'],
                random_state=self.random_state
            )
    
    def objective(self, trial, model_name):
        """Objective function for Optuna optimization"""
        # Sample hyperparameters
        params = {}
        param_space = self.define_hyperparameter_space(model_name)
        
        for param_name, distribution in param_space.items():
            if isinstance(distribution, list):
                # Categorical parameter
                params[param_name] = trial.suggest_categorical(param_name, distribution)
            elif isinstance(distribution, tuple):
                # Numerical parameter
                if isinstance(distribution[0], int):
                    params[param_name] = trial.suggest_int(param_name, distribution[0], distribution[1])
                else:
                    params[param_name] = trial.suggest_float(param_name, distribution[0], distribution[1])
        
        # Create model
        model = self.create_model(model_name, params)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # For tree-based models, we can use early stopping
        if model_name in ["XGBoost", "LightGBM", "CatBoost"]:
            scores = []
            for train_idx, val_idx in cv.split(self.X, self.y):
                X_train, X_val = self.X[train_idx], self.X[val_idx]
                y_train, y_val = self.y[train_idx], self.y[val_idx]
                
                if model_name == "XGBoost":
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
                elif model_name == "LightGBM":
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                elif model_name == "CatBoost":
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
        self.results.append(result)
        
        return mean_score
    
    def optimize_model(self, model_name):
        """Optimize hyperparameters for a specific model"""
        print(f"\n=== OPTIMIZING {model_name.upper()} ===")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
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
        print("=== COMPREHENSIVE HYPERPARAMETER OPTIMIZATION ===")
        
        models = ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "GradientBoosting", "LogisticRegression"]
        studies = {}
        
        for model_name in models:
            study = self.optimize_model(model_name)
            studies[model_name] = study
            
            # Update best model if this is better
            if study.best_value > self.best_score:
                self.best_score = study.best_value
                self.best_params = study.best_params
                self.best_model_name = model_name
                self.best_study = study
        
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
    
    def generate_results_report(self):
        """Generate comprehensive results report"""
        print("\n=== GENERATING RESULTS REPORT ===")
        
        # Create results DataFrame
        results_df = []
        for result in self.results:
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
        results_df.to_csv('hyperparameter_results.csv', index=False)
        best_results.to_csv('best_configurations.csv', index=False)
        
        return results_df, best_results
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n=== CREATING VISUALIZATIONS ===")
        
        # 1. Optimization history for each model
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Hyperparameter Optimization History', fontsize=16, fontweight='bold')
        
        for i, (model_name, study) in enumerate(self.studies.items()):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            trials_df = study.trials_dataframe()
            ax.plot(trials_df['number'], trials_df['value'], 'b-', alpha=0.6)
            ax.set_title(f'{model_name}')
            ax.set_xlabel('Trial')
            ax.set_ylabel('Accuracy')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('optimization_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Best scores comparison
        plt.figure(figsize=(12, 8))
        model_names = list(self.studies.keys())
        best_scores = [study.best_value for study in self.studies.values()]
        
        bars = plt.bar(model_names, best_scores, color='skyblue', alpha=0.7)
        plt.title('Best Cross-Validation Scores by Model', fontsize=14, fontweight='bold')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, best_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('best_scores_comparison.png', dpi=300, bbox_inches='tight')
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
                plt.savefig('parameter_importance.png', dpi=300, bbox_inches='tight')
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
            plt.savefig('feature_importance_optimized.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_best_model(self):
        """Save the best model and all components"""
        print("\n=== SAVING BEST MODEL ===")
        
        # Save model
        joblib.dump(self.best_model, 'best_optimized_model.pkl')
        
        # Save preprocessor
        joblib.dump(self.preprocessor, 'preprocessor_optimized.pkl')
        
        # Save label encoder
        joblib.dump(self.label_encoder, 'label_encoder_optimized.pkl')
        
        # Save feature names
        joblib.dump(self.feature_names, 'feature_names_optimized.pkl')
        
        # Save best parameters
        joblib.dump(self.best_params, 'best_params.pkl')
        
        # Save optimization results
        joblib.dump(self.results, 'optimization_results.pkl')
        
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
        
        submission.to_csv('submission_optimized.csv', index=False)
        
        print("Predictions saved to 'submission_optimized.csv'")
        
        # Show prediction distribution
        pred_counts = submission['Personality'].value_counts()
        print(f"Prediction distribution:")
        for personality, count in pred_counts.items():
            print(f"  {personality}: {count} ({count/len(submission)*100:.1f}%)")
        
        return submission
    
    def run_complete_optimization(self):
        """Run the complete hyperparameter optimization pipeline"""
        print("=== COMPREHENSIVE HYPERPARAMETER OPTIMIZATION PIPELINE ===\n")
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Optimize all models
        self.optimize_all_models()
        
        # Create best model
        self.create_best_model()
        
        # Generate results report
        results_df, best_results = self.generate_results_report()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save best model
        self.save_best_model()
        
        # Generate predictions
        submission = self.generate_predictions()
        
        print("\n=== OPTIMIZATION PIPELINE COMPLETED ===")
        print(f"Best model: {self.best_model_name}")
        print(f"Best cross-validation accuracy: {self.best_score:.4f}")
        print(f"Total trials completed: {len(self.results)}")
        
        return self.best_model, self.best_score, self.best_params

if __name__ == "__main__":
    # Create optimizer
    optimizer = HyperparameterOptimizer(n_trials=50, cv_folds=5, random_state=42)
    
    # Run complete optimization
    best_model, best_score, best_params = optimizer.run_complete_optimization() 
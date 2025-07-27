import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
import joblib
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class PersonalityClassifier:
    def __init__(self):
        self.best_model = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("Loading data...")
        self.train_data = pd.read_csv('train.csv')
        self.test_data = pd.read_csv('test.csv')
        
        print(f"Training data shape: {self.train_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
        
        return self.train_data, self.test_data
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        
        # Basic info
        print("\nDataset Info:")
        print(self.train_data.info())
        
        print("\nMissing values:")
        print(self.train_data.isnull().sum())
        
        print("\nData types:")
        print(self.train_data.dtypes)
        
        # Target distribution
        print("\nTarget variable distribution:")
        target_counts = self.train_data['Personality'].value_counts()
        print(target_counts)
        print(f"Class balance: {target_counts['Extrovert'] / len(self.train_data):.3f} vs {target_counts['Introvert'] / len(self.train_data):.3f}")
        
        # Numerical features summary
        numerical_cols = self.train_data.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in numerical_cols:
            numerical_cols.remove('id')
        
        print(f"\nNumerical features: {numerical_cols}")
        print("\nNumerical features summary:")
        print(self.train_data[numerical_cols].describe())
        
        # Categorical features
        categorical_cols = self.train_data.select_dtypes(include=['object']).columns.tolist()
        if 'Personality' in categorical_cols:
            categorical_cols.remove('Personality')
        
        print(f"\nCategorical features: {categorical_cols}")
        for col in categorical_cols:
            print(f"\n{col} distribution:")
            print(self.train_data[col].value_counts())
        
        # Create visualizations
        self.create_visualizations()
        
    def create_visualizations(self):
        """Create EDA visualizations"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Exploratory Data Analysis', fontsize=16, fontweight='bold')
        
        # Target distribution
        target_counts = self.train_data['Personality'].value_counts()
        axes[0, 0].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Target Distribution')
        
        # Time spent alone vs Personality
        axes[0, 1].boxplot([self.train_data[self.train_data['Personality'] == 'Introvert']['Time_spent_Alone'],
                           self.train_data[self.train_data['Personality'] == 'Extrovert']['Time_spent_Alone']],
                          labels=['Introvert', 'Extrovert'])
        axes[0, 1].set_title('Time Spent Alone vs Personality')
        axes[0, 1].set_ylabel('Hours')
        
        # Social event attendance vs Personality
        axes[0, 2].boxplot([self.train_data[self.train_data['Personality'] == 'Introvert']['Social_event_attendance'],
                           self.train_data[self.train_data['Personality'] == 'Extrovert']['Social_event_attendance']],
                          labels=['Introvert', 'Extrovert'])
        axes[0, 2].set_title('Social Event Attendance vs Personality')
        axes[0, 2].set_ylabel('Frequency')
        
        # Friends circle size vs Personality
        axes[1, 0].boxplot([self.train_data[self.train_data['Personality'] == 'Introvert']['Friends_circle_size'],
                           self.train_data[self.train_data['Personality'] == 'Extrovert']['Friends_circle_size']],
                          labels=['Introvert', 'Extrovert'])
        axes[1, 0].set_title('Friends Circle Size vs Personality')
        axes[1, 0].set_ylabel('Number of Friends')
        
        # Stage fear vs Personality
        stage_fear_cross = pd.crosstab(self.train_data['Stage_fear'], self.train_data['Personality'])
        stage_fear_cross.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Stage Fear vs Personality')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Drained after socializing vs Personality
        drained_cross = pd.crosstab(self.train_data['Drained_after_socializing'], self.train_data['Personality'])
        drained_cross.plot(kind='bar', ax=axes[1, 2])
        axes[1, 2].set_title('Drained After Socializing vs Personality')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def preprocess_data(self):
        """Preprocess the data for modeling"""
        print("\n=== DATA PREPROCESSING ===")
        
        # Separate features and target
        X = self.train_data.drop(['id', 'Personality'], axis=1)
        y = self.train_data['Personality']
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Identify feature types
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numerical features: {numerical_features}")
        print(f"Categorical features: {categorical_features}")
        
        # Create preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
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
        
        # Apply preprocessing
        X_processed = preprocessor.fit_transform(X)
        
        # Get feature names after preprocessing
        numerical_feature_names = numerical_features
        categorical_feature_names = []
        for i, feature in enumerate(categorical_features):
            categories = preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[i][1:]
            categorical_feature_names.extend([f"{feature}_{cat}" for cat in categories])
        
        self.feature_names = numerical_feature_names + categorical_feature_names
        self.preprocessor = preprocessor
        
        print(f"Processed features shape: {X_processed.shape}")
        print(f"Number of features after preprocessing: {len(self.feature_names)}")
        
        return X_processed, y_encoded
    
    def train_models(self, X, y):
        """Train and evaluate multiple models"""
        print("\n=== MODEL TRAINING & EVALUATION ===")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Define models to test
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'CatBoost': cb.CatBoostClassifier(random_state=42, verbose=False)
        }
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Evaluate models
        results = {}
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            mean_cv_score = cv_scores.mean()
            std_cv_score = cv_scores.std()
            
            # Train on full training set
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_pred)
            val_f1 = f1_score(y_val, y_pred, pos_label=1)  # 1 represents Extrovert after encoding
            
            results[name] = {
                'model': model,
                'cv_mean': mean_cv_score,
                'cv_std': std_cv_score,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1
            }
            
            print(f"  CV Accuracy: {mean_cv_score:.4f} (+/- {std_cv_score * 2:.4f})")
            print(f"  Validation Accuracy: {val_accuracy:.4f}")
            print(f"  Validation F1-Score: {val_f1:.4f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best CV Accuracy: {results[best_model_name]['cv_mean']:.4f}")
        
        # Detailed evaluation of best model
        print(f"\nDetailed evaluation of {best_model_name}:")
        y_pred_best = self.best_model.predict(X_val)
        
        # Convert back to original labels for reporting
        y_val_original = self.label_encoder.inverse_transform(y_val)
        y_pred_original = self.label_encoder.inverse_transform(y_pred_best)
        
        print("\nClassification Report:")
        print(classification_report(y_val_original, y_pred_original))
        
        # Confusion matrix
        cm = confusion_matrix(y_val_original, y_pred_original)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Introvert', 'Extrovert'],
                   yticklabels=['Introvert', 'Extrovert'])
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results
    
    def hyperparameter_tuning(self, X, y):
        """Perform hyperparameter tuning on the best model"""
        print("\n=== HYPERPARAMETER TUNING ===")
        
        # Define parameter grids for different models
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 62, 127]
            }
        }
        
        # Get the best model type
        if isinstance(self.best_model, RandomForestClassifier):
            model_name = 'Random Forest'
            base_model = RandomForestClassifier(random_state=42)
        elif isinstance(self.best_model, xgb.XGBClassifier):
            model_name = 'XGBoost'
            base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        elif isinstance(self.best_model, lgb.LGBMClassifier):
            model_name = 'LightGBM'
            base_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        else:
            print("Skipping hyperparameter tuning for this model type")
            return
        
        print(f"Tuning hyperparameters for {model_name}...")
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            base_model,
            param_grids[model_name],
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Update best model
        self.best_model = grid_search.best_estimator_
        
        return grid_search
    
    def feature_importance_analysis(self, X, y):
        """Analyze feature importance"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        if hasattr(self.best_model, 'feature_importances_'):
            # Get feature importances
            importances = self.best_model.feature_importances_
            
            # Create feature importance DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("Top 10 most important features:")
            print(feature_importance_df.head(10))
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = feature_importance_df.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Feature Importances')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # SHAP analysis for tree-based models
            if hasattr(self.best_model, 'predict_proba'):
                try:
                    explainer = shap.TreeExplainer(self.best_model)
                    shap_values = explainer.shap_values(X[:100])  # Use subset for speed
                    
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(shap_values, X[:100], feature_names=self.feature_names, show=False)
                    plt.title('SHAP Feature Importance')
                    plt.tight_layout()
                    plt.savefig('shap_importance.png', dpi=300, bbox_inches='tight')
                    plt.show()
                except Exception as e:
                    print(f"SHAP analysis failed: {e}")
        
        return feature_importance_df if 'feature_importance_df' in locals() else None
    
    def save_model(self):
        """Save the trained model"""
        print("\n=== SAVING MODEL ===")
        
        # Save the best model
        joblib.dump(self.best_model, 'best_personality_model.pkl')
        
        # Save the preprocessor
        joblib.dump(self.preprocessor, 'preprocessor.pkl')
        
        # Save feature names
        joblib.dump(self.feature_names, 'feature_names.pkl')
        
        # Save label encoder
        joblib.dump(self.label_encoder, 'label_encoder.pkl')
        
        print("Model saved as 'best_personality_model.pkl'")
        print("Preprocessor saved as 'preprocessor.pkl'")
        print("Feature names saved as 'feature_names.pkl'")
        print("Label encoder saved as 'label_encoder.pkl'")
    
    def generate_predictions(self):
        """Generate predictions for test set"""
        print("\n=== GENERATING PREDICTIONS ===")
        
        # Prepare test data
        X_test = self.test_data.drop(['id'], axis=1)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Make predictions
        predictions_encoded = self.best_model.predict(X_test_processed)
        
        # Convert back to original labels
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        # Create submission file
        submission = pd.DataFrame({
            'id': self.test_data['id'],
            'Personality': predictions
        })
        
        submission.to_csv('submission.csv', index=False)
        print("Predictions saved to 'submission.csv'")
        
        # Show prediction distribution
        pred_counts = submission['Personality'].value_counts()
        print(f"\nPrediction distribution:")
        print(pred_counts)
        
        return submission
    
    def generate_report(self, results, feature_importance_df):
        """Generate a comprehensive report"""
        print("\n=== GENERATING REPORT ===")
        
        report = f"""
# Personality Classification Model Report

## Dataset Overview
- Training samples: {len(self.train_data)}
- Test samples: {len(self.test_data)}
- Features: {len(self.feature_names)}
- Target: Personality (Extrovert/Introvert)

## Data Quality
- Missing values handled with appropriate imputation strategies
- Categorical variables encoded using one-hot encoding
- Numerical features scaled using StandardScaler

## Model Performance Comparison
"""
        
        for name, result in results.items():
            report += f"""
### {name}
- Cross-validation accuracy: {result['cv_mean']:.4f} (+/- {result['cv_std'] * 2:.4f})
- Validation accuracy: {result['val_accuracy']:.4f}
- Validation F1-score: {result['val_f1']:.4f}
"""
        
        # Best model info
        best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
        report += f"""
## Best Model: {best_model_name}
- Final accuracy: {results[best_model_name]['cv_mean']:.4f}
- Model type: {type(self.best_model).__name__}

## Feature Importance
"""
        
        if feature_importance_df is not None:
            report += "Top 10 most important features:\n"
            for idx, row in feature_importance_df.head(10).iterrows():
                report += f"- {row['feature']}: {row['importance']:.4f}\n"
        
        report += """
## Files Generated
- best_personality_model.pkl: Trained model
- preprocessor.pkl: Data preprocessing pipeline
- feature_names.pkl: Feature names after preprocessing
- submission.csv: Test set predictions
- eda_visualizations.png: Exploratory data analysis plots
- confusion_matrix.png: Model confusion matrix
- feature_importance.png: Feature importance plot
- shap_importance.png: SHAP feature importance (if applicable)

## Usage
```python
import joblib

# Load the model and preprocessor
model = joblib.load('best_personality_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
feature_names = joblib.load('feature_names.pkl')

# Make predictions on new data
X_new_processed = preprocessor.transform(X_new)
predictions = model.predict(X_new_processed)
```
"""
        
        with open('model_report.md', 'w') as f:
            f.write(report)
        
        print("Report saved to 'model_report.md'")
    
    def run_complete_pipeline(self):
        """Run the complete machine learning pipeline"""
        print("=== PERSONALITY CLASSIFICATION PIPELINE ===\n")
        
        # Load data
        self.load_data()
        
        # Explore data
        self.explore_data()
        
        # Preprocess data
        X_processed, y = self.preprocess_data()
        
        # Train models
        results = self.train_models(X_processed, y)
        
        # Hyperparameter tuning
        self.hyperparameter_tuning(X_processed, y)
        
        # Feature importance analysis
        feature_importance_df = self.feature_importance_analysis(X_processed, y)
        
        # Save model
        self.save_model()
        
        # Generate predictions
        submission = self.generate_predictions()
        
        # Generate report
        self.generate_report(results, feature_importance_df)
        
        print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
        print("All files have been generated and saved.")

if __name__ == "__main__":
    # Create and run the classifier
    classifier = PersonalityClassifier()
    classifier.run_complete_pipeline() 
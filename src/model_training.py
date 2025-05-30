"""
Phase 2: Model Training Module
This module handles training the XGBoost classifier for payment risk prediction
"""

import pandas as pd
import numpy as np
import logging
import joblib
import os
import sys
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from typing import Tuple, Dict, Any

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import XGBOOST_PARAMS, DATA_CONFIG, MODELS_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class to handle model training tasks"""
    
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Step 2.1: Split data into train and test sets
        """
        logger.info("Splitting data into train and test sets...")
        
        test_size = DATA_CONFIG['test_size']
        random_state = DATA_CONFIG['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Ensure balanced split
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = list(X.columns)
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        logger.info(f"Train target distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Test target distribution: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def train_xgboost_model(self, X_train: pd.DataFrame = None, y_train: pd.Series = None) -> XGBClassifier:
        """
        Step 2.2: Train XGBoost classifier
        """
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
            
        if X_train is None or y_train is None:
            raise ValueError("Training data not available. Please split data first.")
        
        logger.info("Training XGBoost model...")
        
        # Initialize model with config parameters
        self.model = XGBClassifier(**XGBOOST_PARAMS)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        logger.info("Model training completed!")
        
        # Display feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 Feature Importances:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self.model
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame = None, y_train: pd.Series = None) -> XGBClassifier:
        """
        Optional: Perform hyperparameter tuning using GridSearchCV
        """
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
            
        logger.info("Starting hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Initialize base model
        base_model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        return self.model
    
    def evaluate_model(self, X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Dict[str, Any]:
        """
        Evaluate the trained model
        """
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
            
        if self.model is None:
            raise ValueError("Model not trained. Please train model first.")
        
        logger.info("Evaluating model performance...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=['No Default', 'Default'])
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5, scoring='roc_auc')
        
        metrics = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        # Log results
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        logger.info(f"CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        logger.info(f"Classification Report:\n{class_report}")
        
        return metrics
    
    def save_model(self, filename: str = "xgboost_payment_risk_model.pkl") -> str:
        """
        Save the trained model to file
        """
        if self.model is None:
            raise ValueError("No model to save. Please train model first.")
        
        model_path = os.path.join(MODELS_PATH, filename)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'training_params': XGBOOST_PARAMS
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to: {model_path}")
        
        return model_path
    
    def load_model(self, filename: str = "xgboost_payment_risk_model.pkl") -> XGBClassifier:
        """
        Load a trained model from file
        """
        model_path = os.path.join(MODELS_PATH, filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        
        logger.info(f"Model loaded from: {model_path}")
        return self.model
    
    def train_complete_pipeline(self, X: pd.DataFrame, y: pd.Series, tune_hyperparameters: bool = False) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        """
        logger.info("Starting complete model training pipeline...")
        
        # Step 2.1: Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Step 2.2: Train model
        if tune_hyperparameters:
            model = self.hyperparameter_tuning(X_train, y_train)
        else:
            model = self.train_xgboost_model(X_train, y_train)
        
        # Evaluate model
        metrics = self.evaluate_model(X_test, y_test)
        
        # Save model
        model_path = self.save_model()
        
        logger.info("Model training pipeline completed successfully!")
        
        return {
            'model': model,
            'metrics': metrics,
            'model_path': model_path,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }


# Legacy function for backward compatibility
def train_model(customer_df):
    """Train XGBoost model (legacy function)"""
    X = customer_df.drop(columns=['未払FLAG', 'レコード番号'])  # Features
    y = customer_df['未払FLAG']  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    return model, X_test, y_test


def main():
    """Main function for testing model training"""
    # This would typically be called after data preparation
    from data_preparation import DataPreparator
    
    # Prepare data
    preparator = DataPreparator()
    X, y = preparator.run_full_preparation()
    
    # Train model
    trainer = ModelTrainer()
    results = trainer.train_complete_pipeline(X, y, tune_hyperparameters=False)
    
    print("\n=== Model Training Summary ===")
    print(f"Model: {type(results['model']).__name__}")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"ROC AUC: {results['metrics']['roc_auc']:.4f}")
    print(f"Model saved to: {results['model_path']}")


if __name__ == "__main__":
    main()
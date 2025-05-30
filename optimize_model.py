#!/usr/bin/env python3
"""
Model Performance Optimization Script

This script implements various techniques to improve the XGBoost model performance,
particularly focusing on addressing the low ROC AUC score.
"""

import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE

from data_preparation import DataPreparator
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Enhanced model training with optimization techniques."""
    
    def __init__(self):
        self.data_prep = DataPreparator()
        self.best_model = None
        self.best_score = 0
        self.optimization_results = {}
    
    def load_and_prepare_data(self):
        """Load and prepare data for optimization."""
        logger.info("🔄 Loading data for optimization...")
        
        # Use the existing data preparation pipeline
        self.data_prep.run_full_pipeline()
        X, y = self.data_prep.prepare_ml_data()
        
        logger.info(f"📊 Dataset shape: {X.shape}")
        logger.info(f"🎯 Target distribution: {dict(pd.Series(y).value_counts())}")
        
        return X, y
    
    def apply_smote_balancing(self, X, y):
        """Apply SMOTE to balance the dataset."""
        logger.info("⚖️ Applying SMOTE balancing...")
        
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        logger.info(f"📊 Original shape: {X.shape}")
        logger.info(f"📊 Balanced shape: {X_balanced.shape}")
        logger.info(f"🎯 New target distribution: {dict(pd.Series(y_balanced).value_counts())}")
        
        return X_balanced, y_balanced
    
    def feature_selection(self, X, y, k=15):
        """Select top k features using statistical methods."""
        logger.info(f"🎯 Selecting top {k} features...")
        
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        feature_mask = selector.get_support()
        selected_features = [X.columns[i] for i in range(len(X.columns)) if feature_mask[i]]
        
        logger.info(f"✅ Selected features: {selected_features}")
        
        return pd.DataFrame(X_selected, columns=selected_features), selected_features
    
    def optimize_xgboost_hyperparameters(self, X, y):
        """Optimize XGBoost hyperparameters using GridSearchCV."""
        logger.info("🔧 Optimizing XGBoost hyperparameters...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }
        
        # Use smaller grid for faster execution
        quick_param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [4, 6],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Use stratified k-fold for imbalanced dataset
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=quick_param_grid,
            scoring='roc_auc',
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"🎯 Best ROC AUC: {grid_search.best_score_:.4f}")
        logger.info(f"🔧 Best parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_
    
    def try_alternative_algorithms(self, X, y):
        """Try alternative machine learning algorithms."""
        logger.info("🔄 Testing alternative algorithms...")
        
        algorithms = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=42,
                class_weight='balanced'
            ),
            'XGBoost Balanced': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=4.75,  # ratio of negative to positive samples
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        }
        
        results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in algorithms.items():
            # Cross-validation scores
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            
            # Fit and predict for detailed metrics
            model.fit(X, y)
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]
            
            roc_auc = roc_auc_score(y, y_proba)
            
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'roc_auc': roc_auc,
                'cv_scores': cv_scores
            }
            
            logger.info(f"📊 {name}: ROC AUC = {roc_auc:.4f} (CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f})")
        
        return results
    
    def run_complete_optimization(self):
        """Run the complete optimization pipeline."""
        logger.info("🚀 Starting Model Optimization Pipeline")
        logger.info("=" * 60)
        
        # 1. Load and prepare data
        X, y = self.load_and_prepare_data()
        original_X, original_y = X.copy(), y.copy()
        
        results = {}
        
        # 2. Baseline model (current)
        logger.info("\n📊 BASELINE MODEL PERFORMANCE")
        logger.info("-" * 40)
        
        baseline_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        baseline_model.fit(X, y)
        baseline_proba = baseline_model.predict_proba(X)[:, 1]
        baseline_roc = roc_auc_score(y, baseline_proba)
        
        results['baseline'] = {
            'model': baseline_model,
            'roc_auc': baseline_roc,
            'description': 'Original XGBoost model'
        }
        
        logger.info(f"📈 Baseline ROC AUC: {baseline_roc:.4f}")
        
        # 3. Feature selection optimization
        logger.info("\n🎯 FEATURE SELECTION OPTIMIZATION")
        logger.info("-" * 40)
        
        X_selected, selected_features = self.feature_selection(X, y, k=15)
        
        fs_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        fs_model.fit(X_selected, y)
        fs_proba = fs_model.predict_proba(X_selected)[:, 1]
        fs_roc = roc_auc_score(y, fs_proba)
        
        results['feature_selection'] = {
            'model': fs_model,
            'roc_auc': fs_roc,
            'features': selected_features,
            'description': f'XGBoost with top {len(selected_features)} features'
        }
        
        logger.info(f"📈 Feature Selection ROC AUC: {fs_roc:.4f}")
        
        # 4. SMOTE balancing
        logger.info("\n⚖️ SMOTE BALANCING OPTIMIZATION")
        logger.info("-" * 40)
        
        X_balanced, y_balanced = self.apply_smote_balancing(X, y)
        
        smote_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        smote_model.fit(X_balanced, y_balanced)
        smote_proba = smote_model.predict_proba(X)[:, 1]  # Predict on original data
        smote_roc = roc_auc_score(y, smote_proba)
        
        results['smote'] = {
            'model': smote_model,
            'roc_auc': smote_roc,
            'description': 'XGBoost with SMOTE balancing'
        }
        
        logger.info(f"📈 SMOTE ROC AUC: {smote_roc:.4f}")
        
        # 5. Hyperparameter optimization
        logger.info("\n🔧 HYPERPARAMETER OPTIMIZATION")
        logger.info("-" * 40)
        
        try:
            best_model, best_score, best_params = self.optimize_xgboost_hyperparameters(X, y)
            
            results['hyperparameter_tuned'] = {
                'model': best_model,
                'roc_auc': best_score,
                'params': best_params,
                'description': 'XGBoost with optimized hyperparameters'
            }
            
            logger.info(f"📈 Hyperparameter Tuned ROC AUC: {best_score:.4f}")
        except Exception as e:
            logger.warning(f"⚠️ Hyperparameter optimization failed: {e}")
        
        # 6. Alternative algorithms
        logger.info("\n🔄 ALTERNATIVE ALGORITHMS")
        logger.info("-" * 40)
        
        alt_results = self.try_alternative_algorithms(X, y)
        results.update(alt_results)
        
        # 7. Combined approach (SMOTE + Feature Selection)
        logger.info("\n🚀 COMBINED OPTIMIZATION")
        logger.info("-" * 40)
        
        X_balanced_selected, _ = self.feature_selection(
            pd.DataFrame(X_balanced, columns=X.columns), 
            y_balanced, 
            k=15
        )
        
        combined_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        combined_model.fit(X_balanced_selected, y_balanced)
        
        # Predict on original data with selected features
        X_original_selected = X[X_balanced_selected.columns]
        combined_proba = combined_model.predict_proba(X_original_selected)[:, 1]
        combined_roc = roc_auc_score(y, combined_proba)
        
        results['combined'] = {
            'model': combined_model,
            'roc_auc': combined_roc,
            'description': 'SMOTE + Feature Selection + XGBoost'
        }
        
        logger.info(f"📈 Combined Approach ROC AUC: {combined_roc:.4f}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("🏆 OPTIMIZATION RESULTS SUMMARY")
        logger.info("=" * 60)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
        
        for i, (name, result) in enumerate(sorted_results, 1):
            logger.info(f"{i}. {name:20s}: ROC AUC = {result['roc_auc']:.4f} - {result['description']}")
        
        # Find best model
        best_approach = sorted_results[0]
        self.best_model = best_approach[1]['model']
        self.best_score = best_approach[1]['roc_auc']
        
        logger.info(f"\n🎯 Best approach: {best_approach[0]} with ROC AUC = {self.best_score:.4f}")
        
        improvement = ((self.best_score - baseline_roc) / baseline_roc) * 100
        logger.info(f"📈 Improvement over baseline: {improvement:+.2f}%")
        
        self.optimization_results = results
        return results

def main():
    """Main execution function."""
    optimizer = ModelOptimizer()
    results = optimizer.run_complete_optimization()
    
    print("\n" + "="*60)
    print("✅ MODEL OPTIMIZATION COMPLETED")
    print("="*60)
    print(f"📊 Best ROC AUC achieved: {optimizer.best_score:.4f}")
    print("📁 Check logs above for detailed analysis")
    print("💡 Consider implementing the best performing approach in your main pipeline")

if __name__ == "__main__":
    main()

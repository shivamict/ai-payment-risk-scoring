"""
Phase 3: Scoring & Explanation Module
This module handles risk score generation and SHAP explanations
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import SHAP, use fallback if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available. Using fallback explanation methods.")

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SCORING_CONFIG, OUTPUT_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskScorer:
    """Class to handle risk scoring and explanation tasks"""
    
    def __init__(self, model=None):
        self.model = model
        self.explainer = None
        self.shap_values = None
        
    def generate_risk_scores(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Step 3.1: Generate risk scores from model predictions
        """
        if self.model is None:
            raise ValueError("Model not provided. Please set model first.")
        
        logger.info("Generating risk scores...")
        
        # Get prediction probabilities
        probs = self.model.predict_proba(X_test)[:, 1]  # Probability of default (class 1)
        
        # Convert to risk scores (0-100 scale, where 100 = low risk, 0 = high risk)
        risk_scores = (1 - probs) * 100
        
        logger.info(f"Risk scores generated for {len(risk_scores)} customers")
        logger.info(f"Risk score distribution:")
        logger.info(f"  Mean: {risk_scores.mean():.2f}")
        logger.info(f"  Std: {risk_scores.std():.2f}")
        logger.info(f"  Min: {risk_scores.min():.2f}")
        logger.info(f"  Max: {risk_scores.max():.2f}")
        
        return risk_scores
    
    def categorize_risk_levels(self, risk_scores: np.ndarray) -> List[str]:
        """
        Categorize risk scores into risk levels
        """
        risk_categories = []
        thresholds = SCORING_CONFIG['risk_thresholds']
        
        for score in risk_scores:
            if score >= thresholds['low']:
                risk_categories.append('Low Risk')
            elif score >= thresholds['medium']:
                risk_categories.append('Medium Risk')
            else:
                risk_categories.append('High Risk')
        
        return risk_categories
    
    def explain_with_shap(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Any:
        """
        Step 3.2: Generate SHAP explanations (or fallback explanations)
        """
        if self.model is None:
            raise ValueError("Model not provided. Please set model first.")
        
        if SHAP_AVAILABLE:
            logger.info("Generating SHAP explanations...")
            
            # Create SHAP explainer
            self.explainer = shap.Explainer(self.model, X_train)
            
            # Calculate SHAP values
            self.shap_values = self.explainer(X_test)
            
            logger.info(f"SHAP explanations generated for {len(X_test)} samples")
            
            return self.shap_values
        else:
            logger.info("Using fallback feature importance for explanations...")
            
            # Fallback: Use feature importance from the model
            feature_importance = self.model.feature_importances_
            
            # Create a simple explanation structure
            self.shap_values = {
                'values': np.array([feature_importance] * len(X_test)),
                'feature_names': X_test.columns.tolist(),
                'type': 'fallback'
            }
            
            logger.info(f"Fallback explanations generated for {len(X_test)} samples")
            
            return self.shap_values
    
    def get_top_risk_factors(self, shap_values: Any = None, X_test: pd.DataFrame = None, top_n: int = 3) -> List[str]:
        """
        Get top risk factors for each customer
        """
        if shap_values is None:
            shap_values = self.shap_values
        if shap_values is None:
            raise ValueError("Explanations not available. Please run explain_with_shap first.")
        
        logger.info(f"Extracting top {top_n} risk factors for each customer...")
        
        top_features = []
        
        if SHAP_AVAILABLE and hasattr(shap_values, 'values'):
            # Original SHAP implementation
            for i in range(len(shap_values)):
                # Get indices of top contributing features (by absolute SHAP value)
                abs_shap_values = np.abs(shap_values[i].values)
                top_idx = abs_shap_values.argsort()[-top_n:][::-1]
                
                # Get feature names and their SHAP values
                feature_contributions = []
                for idx in top_idx:
                    feature_name = X_test.columns[idx]
                    shap_val = shap_values[i].values[idx]
                    contribution_type = "increases" if shap_val > 0 else "decreases"
                    feature_contributions.append(f"{feature_name} ({contribution_type} risk)")
                
                top_features.append(", ".join(feature_contributions))
        else:
            # Fallback implementation using feature importance
            feature_importance = shap_values['values'][0]  # Same for all samples in fallback
            top_idx = feature_importance.argsort()[-top_n:][::-1]
            
            for i in range(len(X_test)):
                feature_contributions = []
                for idx in top_idx:
                    feature_name = shap_values['feature_names'][idx]
                    feature_contributions.append(f"{feature_name} (important feature)")
                
                top_features.append(", ".join(feature_contributions))
        
        return top_features
    
    def create_shap_plots(self, X_test: pd.DataFrame, save_plots: bool = True) -> Dict[str, str]:
        """
        Create and save SHAP visualization plots (or fallback plots)
        """
        if self.shap_values is None:
            raise ValueError("Explanations not available. Please run explain_with_shap first.")
        
        logger.info("Creating visualization plots...")
        
        plot_paths = {}
        
        try:
            if SHAP_AVAILABLE and hasattr(self.shap_values, 'values'):
                # Original SHAP plots
                # 1. Summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(self.shap_values, X_test, show=False)
                if save_plots:
                    summary_path = os.path.join(OUTPUT_DIR, "shap_summary_plot.png")
                    plt.savefig(summary_path, bbox_inches='tight', dpi=300)
                    plot_paths['summary'] = summary_path
                    logger.info(f"SHAP summary plot saved to: {summary_path}")
                plt.close()
            else:
                # Fallback: Feature importance plot
                plt.figure(figsize=(12, 8))
                feature_importance = self.shap_values['values'][0]
                feature_names = self.shap_values['feature_names']
                
                # Create feature importance plot
                indices = np.argsort(feature_importance)
                plt.figure(figsize=(10, 8))
                plt.barh(range(len(feature_importance)), feature_importance[indices])
                plt.yticks(range(len(feature_importance)), [feature_names[i] for i in indices])
                plt.xlabel('Feature Importance')
                plt.title('Feature Importance (Fallback Explanation)')
                plt.tight_layout()
                
                if save_plots:
                    summary_path = os.path.join(OUTPUT_DIR, "feature_importance_plot.png")
                    plt.savefig(summary_path, bbox_inches='tight', dpi=300)
                    plot_paths['summary'] = summary_path
                    logger.info(f"Feature importance plot saved to: {summary_path}")
                plt.close()
            
                # Additional SHAP plots for the original implementation
                if SHAP_AVAILABLE and hasattr(self.shap_values, 'values'):
                    # 2. Feature importance plot
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(self.shap_values, X_test, plot_type="bar", show=False)
                    if save_plots:
                        importance_path = os.path.join(OUTPUT_DIR, "shap_feature_importance.png")
                        plt.savefig(importance_path, bbox_inches='tight', dpi=300)
                        plot_paths['importance'] = importance_path
                        logger.info(f"SHAP feature importance plot saved to: {importance_path}")
                    plt.close()
                    
                    # 3. Waterfall plot for first customer (example)
                    if len(self.shap_values) > 0:
                        plt.figure(figsize=(10, 6))
                        shap.plots.waterfall(self.shap_values[0], show=False)
                        if save_plots:
                            waterfall_path = os.path.join(OUTPUT_DIR, "shap_waterfall_example.png")
                            plt.savefig(waterfall_path, bbox_inches='tight', dpi=300)
                            plot_paths['waterfall'] = waterfall_path
                            logger.info(f"SHAP waterfall plot saved to: {waterfall_path}")
                        plt.close()
            
        except Exception as e:
            logger.error(f"Error creating visualization plots: {str(e)}")
        
        return plot_paths
    
    def create_risk_distribution_plot(self, risk_scores: np.ndarray, save_plot: bool = True) -> str:
        """
        Create risk score distribution plot
        """
        logger.info("Creating risk score distribution plot...")
        
        plt.figure(figsize=(12, 6))
        
        # Create subplot layout
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram of risk scores
        ax1.hist(risk_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Risk Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Risk Scores')
        ax1.grid(True, alpha=0.3)
        
        # Risk categories pie chart
        risk_categories = self.categorize_risk_levels(risk_scores)
        category_counts = pd.Series(risk_categories).value_counts()
        
        colors = ['green', 'orange', 'red']
        ax2.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', colors=colors)
        ax2.set_title('Risk Level Distribution')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(OUTPUT_DIR, "risk_score_distribution.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            logger.info(f"Risk distribution plot saved to: {plot_path}")
            plt.close()
            return plot_path
        
        return None
    
    def generate_customer_explanations(self, customer_idx: int, X_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate detailed explanation for a specific customer
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not available. Please run explain_with_shap first.")
        
        if customer_idx >= len(self.shap_values):
            raise ValueError(f"Customer index {customer_idx} out of range")
        
        customer_shap = self.shap_values[customer_idx]
        customer_features = X_test.iloc[customer_idx]
        
        # Get feature contributions
        feature_contributions = []
        for i, feature in enumerate(X_test.columns):
            contribution = {
                'feature': feature,
                'value': customer_features[feature],
                'shap_value': customer_shap.values[i],
                'impact': 'Increases Risk' if customer_shap.values[i] > 0 else 'Decreases Risk'
            }
            feature_contributions.append(contribution)
        
        # Sort by absolute SHAP value
        feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        return {
            'customer_index': customer_idx,
            'feature_contributions': feature_contributions,
            'base_value': customer_shap.base_values,
            'expected_value': self.explainer.expected_value
        }
    
    def run_complete_scoring(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series = None) -> Dict[str, Any]:
        """
        Run the complete scoring and explanation pipeline
        """
        logger.info("Starting complete scoring pipeline...")
        
        # Step 3.1: Generate risk scores
        risk_scores = self.generate_risk_scores(X_test)
        
        # Categorize risk levels
        risk_categories = self.categorize_risk_levels(risk_scores)
        
        # Step 3.2: Generate SHAP explanations
        shap_values = self.explain_with_shap(X_train, X_test)
        
        # Get top risk factors
        top_risk_factors = self.get_top_risk_factors(shap_values, X_test)
        
        # Create visualizations
        shap_plots = self.create_shap_plots(X_test)
        risk_plot = self.create_risk_distribution_plot(risk_scores)
        
        # Compile results
        results = {
            'risk_scores': risk_scores,
            'risk_categories': risk_categories,
            'top_risk_factors': top_risk_factors,
            'shap_values': shap_values,
            'shap_plots': shap_plots,
            'risk_distribution_plot': risk_plot,
            'X_test': X_test
        }
        
        if y_test is not None:
            results['y_test'] = y_test
        
        logger.info("Scoring pipeline completed successfully!")
        
        return results


# Legacy functions for backward compatibility
def generate_risk_scores(model, X_test):
    """Generate risk scores (legacy function)"""
    probs = model.predict_proba(X_test)[:, 1]  # Probabilities for class "1" = likely to default
    risk_scores = (1 - probs) * 100  # 100 = low risk, 0 = high risk
    return risk_scores

def explain_risk_scores(model, X_train, X_test):
    """Explain risk scores with SHAP (legacy function)"""
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    return shap_values

def get_top_risk_factors(shap_values, X_test, top_n=3):
    """Get top risk factors (legacy function)"""
    top_features = []
    for i in range(len(shap_values)):
        top_idx = shap_values[i].values.argsort()[-top_n:][::-1]
        top_feats = [X_test.columns[j] for j in top_idx]
        top_features.append(", ".join(top_feats))
    return top_features


def main():
    """Main function for testing scoring"""
    # This would typically be called after model training
    from data_preparation import DataPreparator
    from model_training import ModelTrainer
    
    # Prepare data and train model
    preparator = DataPreparator()
    X, y = preparator.run_full_preparation()
    
    trainer = ModelTrainer()
    training_results = trainer.train_complete_pipeline(X, y)
    
    # Score and explain
    scorer = RiskScorer(training_results['model'])
    scoring_results = scorer.run_complete_scoring(
        training_results['X_train'],
        training_results['X_test'],
        training_results['y_test']
    )
    
    print("\n=== Scoring Summary ===")
    print(f"Risk scores generated for {len(scoring_results['risk_scores'])} customers")
    print(f"Average risk score: {scoring_results['risk_scores'].mean():.2f}")
    print(f"Risk categories distribution:")
    categories_count = pd.Series(scoring_results['risk_categories']).value_counts()
    for category, count in categories_count.items():
        print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
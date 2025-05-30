"""
Main Execution Script for AI-based Customer Payment Risk Scoring System
This script orchestrates the complete pipeline from Phase 1 to Phase 5
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from data_preparation import DataPreparator
from model_training import ModelTrainer
from scoring import RiskScorer
from utils import ResultsExporter

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'pipeline_execution.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PaymentRiskPipeline:
    """Complete pipeline for payment risk scoring"""
    
    def __init__(self, excel_file_path: str = None):
        self.excel_file_path = excel_file_path
        self.data_preparator = DataPreparator(excel_file_path)
        self.model_trainer = ModelTrainer()
        self.risk_scorer = None
        self.results_exporter = ResultsExporter()
        
        # Results storage
        self.processed_data = None
        self.X = None
        self.y = None
        self.training_results = None
        self.scoring_results = None
        
    def run_phase_1_data_preparation(self) -> tuple:
        """
        🟦 PHASE 1: Data Preparation
        Clean, process, and engineer features
        """
        logger.info("="*60)
        logger.info("🟦 PHASE 1: Data Preparation")
        logger.info("="*60)
        
        # Run complete data preparation pipeline
        X, y = self.data_preparator.run_full_preparation(self.excel_file_path)
        
        self.X = X
        self.y = y
        self.processed_data = self.data_preparator.processed_data
        
        logger.info("✅ Phase 1 completed successfully!")
        return X, y
    
    def run_phase_2_model_training(self, tune_hyperparameters: bool = False) -> dict:
        """
        🟨 PHASE 2: Model Training (Risk Score)
        Train a machine learning model to predict payment risk
        """
        logger.info("="*60)
        logger.info("🟨 PHASE 2: Model Training")
        logger.info("="*60)
        
        if self.X is None or self.y is None:
            raise ValueError("Data not prepared. Please run Phase 1 first.")
        
        # Train complete model pipeline
        training_results = self.model_trainer.train_complete_pipeline(
            self.X, self.y, tune_hyperparameters=tune_hyperparameters
        )
        
        self.training_results = training_results
        
        logger.info("✅ Phase 2 completed successfully!")
        return training_results
    
    def run_phase_3_scoring_explanation(self) -> dict:
        """
        🟩 PHASE 3: Score & Reasoning
        Generate a score + explain with SHAP
        """
        logger.info("="*60)
        logger.info("🟩 PHASE 3: Scoring & Explanation")
        logger.info("="*60)
        
        if self.training_results is None:
            raise ValueError("Model not trained. Please run Phase 2 first.")
        
        # Initialize risk scorer with trained model
        self.risk_scorer = RiskScorer(self.training_results['model'])
        
        # Run complete scoring pipeline
        scoring_results = self.risk_scorer.run_complete_scoring(
            self.training_results['X_train'],
            self.training_results['X_test'],
            self.training_results['y_test']
        )
        
        self.scoring_results = scoring_results
        
        logger.info("✅ Phase 3 completed successfully!")
        return scoring_results
    
    def run_phase_4_evaluation(self) -> dict:
        """
        🟨 PHASE 4: Evaluation
        Validate model performance and tune it
        """
        logger.info("="*60)
        logger.info("🟨 PHASE 4: Model Evaluation")
        logger.info("="*60)
        
        if self.training_results is None:
            raise ValueError("Model not trained. Please run Phase 2 first.")
        
        # Model evaluation was already done in training, but let's create additional analysis
        evaluation_results = self.create_detailed_evaluation()
        
        logger.info("✅ Phase 4 completed successfully!")
        return evaluation_results
    
    def run_phase_5_output_export(self) -> dict:
        """
        🟦 PHASE 5: Output
        Export score and reasons per customer to CSV or dashboard
        """
        logger.info("="*60)
        logger.info("🟦 PHASE 5: Output & Export")
        logger.info("="*60)
        
        if self.scoring_results is None:
            raise ValueError("Scoring not completed. Please run Phase 3 first.")
        
        # Prepare data for export
        results_df = pd.DataFrame({
            'customer_id': range(len(self.scoring_results['risk_scores'])),
            'risk_score': self.scoring_results['risk_scores'],
            'risk_category': self.scoring_results['risk_categories'],
            'top_risk_factors': self.scoring_results['top_risk_factors']
        })
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': self.training_results['X_train'].columns,
            'importance': self.training_results['model'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Compile pipeline results
        pipeline_results = {
            'training_results': self.training_results,
            'scoring_results': self.scoring_results,
            'processed_data': self.processed_data,
            'evaluation_results': self.create_detailed_evaluation()
        }
        
        # Export results
        export_results = self.results_exporter.export_complete_results(
            results_df,
            feature_importance,
            self.training_results['metrics'],
            pipeline_results
        )
        
        logger.info("✅ Phase 5 completed successfully!")
        return export_results
    
    def create_detailed_evaluation(self) -> dict:
        """Create detailed model evaluation"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report
        )
        
        model = self.training_results['model']
        X_test = self.training_results['X_test']
        y_test = self.training_results['y_test']
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Detailed metrics
        evaluation_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': self.training_results['X_test'].columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        evaluation_metrics['feature_importance'] = feature_importance.to_dict('records')
        
        # Log key metrics
        logger.info(f"Model Performance Summary:")
        logger.info(f"  Accuracy: {evaluation_metrics['accuracy']:.4f}")
        logger.info(f"  ROC AUC: {evaluation_metrics['roc_auc']:.4f}")
        logger.info(f"  Precision: {evaluation_metrics['precision']:.4f}")
        logger.info(f"  Recall: {evaluation_metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {evaluation_metrics['f1_score']:.4f}")
        
        return evaluation_metrics
    
    def run_complete_pipeline(self, tune_hyperparameters: bool = False) -> dict:
        """
        Execute the complete 5-phase pipeline
        """
        start_time = datetime.now()
        
        logger.info("🚀 Starting AI-based Customer Payment Risk Scoring Pipeline")
        logger.info("="*80)
        
        try:
            # Phase 1: Data Preparation
            X, y = self.run_phase_1_data_preparation()
            
            # Phase 2: Model Training
            training_results = self.run_phase_2_model_training(tune_hyperparameters)
            
            # Phase 3: Scoring & Explanation
            scoring_results = self.run_phase_3_scoring_explanation()
            
            # Phase 4: Evaluation
            evaluation_results = self.run_phase_4_evaluation()
            
            # Phase 5: Output & Export
            export_results = self.run_phase_5_output_export()
            
            end_time = datetime.now()
            execution_time = end_time - start_time
            
            # Compile final results
            final_results = {
                'execution_time': str(execution_time),
                'data_shape': {'features': X.shape[1], 'samples': X.shape[0]},
                'model_performance': {
                    'accuracy': training_results['metrics']['accuracy'],
                    'roc_auc': training_results['metrics']['roc_auc']
                },
                'risk_scoring': {
                    'customers_scored': len(scoring_results['risk_scores']),
                    'average_risk_score': float(scoring_results['risk_scores'].mean()),
                    'risk_distribution': pd.Series(scoring_results['risk_categories']).value_counts().to_dict()
                },
                'export_files': export_results
            }
            
            logger.info("="*80)
            logger.info("🎉 Pipeline completed successfully!")
            logger.info(f"⏱️  Total execution time: {execution_time}")
            logger.info(f"📊 Customers scored: {final_results['risk_scoring']['customers_scored']}")
            logger.info(f"🎯 Model accuracy: {final_results['model_performance']['accuracy']:.4f}")
            logger.info(f"📈 Model ROC AUC: {final_results['model_performance']['roc_auc']:.4f}")
            logger.info("="*80)
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            raise


def main(excel_file_path: str = None, tune_hyperparameters: bool = False):
    """
    Main function to execute the complete pipeline
    
    Args:
        excel_file_path: Path to Excel file with customer data
        tune_hyperparameters: Whether to perform hyperparameter tuning
    """
    
    # Initialize and run pipeline
    pipeline = PaymentRiskPipeline(excel_file_path)
    results = pipeline.run_complete_pipeline(tune_hyperparameters)
    
    print("\n" + "="*60)
    print("🏆 AI PAYMENT RISK SCORING - EXECUTION SUMMARY")
    print("="*60)
    print(f"⏱️  Execution Time: {results['execution_time']}")
    print(f"📊 Dataset: {results['data_shape']['samples']} customers, {results['data_shape']['features']} features")
    print(f"🎯 Model Accuracy: {results['model_performance']['accuracy']:.1%}")
    print(f"📈 Model ROC AUC: {results['model_performance']['roc_auc']:.3f}")
    print(f"🔍 Customers Scored: {results['risk_scoring']['customers_scored']}")
    print(f"📊 Average Risk Score: {results['risk_scoring']['average_risk_score']:.1f}/100")
    print("\n📋 Risk Distribution:")
    for risk_level, count in results['risk_scoring']['risk_distribution'].items():
        percentage = (count / results['risk_scoring']['customers_scored']) * 100
        print(f"   {risk_level}: {count} customers ({percentage:.1f}%)")
    
    print("\n📁 Generated Files:")
    for file_type, file_path in results['export_files'].items():
        if file_path:
            print(f"   {file_type}: {file_path}")
    
    print("="*60)
    print("✅ Pipeline completed successfully!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AI-based Customer Payment Risk Scoring System')
    parser.add_argument('--excel_file', type=str, help='Path to Excel file with customer data')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    
    args = parser.parse_args()
    
    main(excel_file_path=args.excel_file, tune_hyperparameters=args.tune)
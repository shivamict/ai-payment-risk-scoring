"""
Utility functions and classes for the AI Payment Risk Scoring System.
Includes data handling, visualization, and Streamlit dashboard functionality.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
import logging
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)
from typing import Dict, List, Any, Optional
import joblib


class ResultsExporter:
    """
    Handles exporting results to various formats including CSV, Excel, and visualizations.
    Implements Phase 5: Output/Export functionality.
    """
    
    def __init__(self, output_dir: str = "output"):
        """Initialize the results exporter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def export_risk_scores(self, results_df: pd.DataFrame, filename: str = None) -> str:
        """Export risk scores to CSV file."""
        if filename is None:
            filename = f"risk_scores_{self.timestamp}.csv"
        
        filepath = self.output_dir / filename
        results_df.to_csv(filepath, index=False)
        return str(filepath)
    
    def export_model_metrics(self, metrics: Dict[str, Any], filename: str = None) -> str:
        """Export model evaluation metrics to JSON and CSV."""
        if filename is None:
            filename = f"model_metrics_{self.timestamp}"
        
        # Export as JSON
        import json
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Export as CSV for easy viewing
        csv_path = self.output_dir / f"{filename}.csv"
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(csv_path, index=False)
        
        return str(json_path)
    
    def export_feature_importance(self, feature_importance: pd.DataFrame, filename: str = None) -> str:
        """Export feature importance to CSV."""
        if filename is None:
            filename = f"feature_importance_{self.timestamp}.csv"
        
        filepath = self.output_dir / filename
        feature_importance.to_csv(filepath, index=False)
        return str(filepath)
    
    def create_risk_distribution_plot(self, results_df: pd.DataFrame, save_path: str = None) -> str:
        """Create and save risk score distribution plot."""
        plt.figure(figsize=(12, 8))
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Risk score distribution
        ax1.hist(results_df['risk_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Risk Score Distribution')
        ax1.set_xlabel('Risk Score')
        ax1.set_ylabel('Frequency')
        
        # Risk category distribution
        risk_counts = results_df['risk_category'].value_counts()
        ax2.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Risk Category Distribution')
        
        # Risk score by payment failure
        if 'payment_failure' in results_df.columns:
            results_df.boxplot(column='risk_score', by='payment_failure', ax=ax3)
            ax3.set_title('Risk Score by Payment Failure')
            ax3.set_xlabel('Payment Failure')
            ax3.set_ylabel('Risk Score')
        
        # Correlation heatmap (if numeric columns exist)
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = results_df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax4)
            ax4.set_title('Feature Correlation Heatmap')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"risk_analysis_{self.timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_summary_report(self, pipeline_results: Dict[str, Any], filename: str = None) -> str:
        """Create a comprehensive summary report."""
        if filename is None:
            filename = f"summary_report_{self.timestamp}.html"
        
        report_path = self.output_dir / filename
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Payment Risk Scoring - Summary Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 10px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI Payment Risk Scoring System - Summary Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Execution Summary</h2>
                <div class="metric">
                    <strong>Total Execution Time:</strong> {pipeline_results.get('total_time', 'N/A')}
                </div>
                <div class="metric">
                    <strong>Customers Processed:</strong> {pipeline_results.get('total_customers', 'N/A')}
                </div>
                <div class="metric">
                    <strong>High Risk Customers:</strong> {pipeline_results.get('high_risk_count', 'N/A')}
                </div>
            </div>
            
            <div class="section">
                <h2>Model Performance</h2>
                <div class="metric">
                    <strong>Accuracy:</strong> {pipeline_results.get('accuracy', 'N/A') if isinstance(pipeline_results.get('accuracy'), str) else f"{pipeline_results.get('accuracy', 0):.4f}"}
                </div>
                <div class="metric">
                    <strong>Precision:</strong> {pipeline_results.get('precision', 'N/A') if isinstance(pipeline_results.get('precision'), str) else f"{pipeline_results.get('precision', 0):.4f}"}
                </div>
                <div class="metric">
                    <strong>Recall:</strong> {pipeline_results.get('recall', 'N/A') if isinstance(pipeline_results.get('recall'), str) else f"{pipeline_results.get('recall', 0):.4f}"}
                </div>
                <div class="metric">
                    <strong>F1-Score:</strong> {pipeline_results.get('f1_score', 'N/A') if isinstance(pipeline_results.get('f1_score'), str) else f"{pipeline_results.get('f1_score', 0):.4f}"}
                </div>
                <div class="metric">
                    <strong>ROC-AUC:</strong> {pipeline_results.get('roc_auc', 'N/A') if isinstance(pipeline_results.get('roc_auc'), str) else f"{pipeline_results.get('roc_auc', 0):.4f}"}
                </div>
            </div>
            
            <div class="section">
                <h2>Files Generated</h2>
                <ul>
                    <li>Risk Scores: {pipeline_results.get('scores_file', 'N/A')}</li>
                    <li>Model Metrics: {pipeline_results.get('metrics_file', 'N/A')}</li>
                    <li>Feature Importance: {pipeline_results.get('importance_file', 'N/A')}</li>
                    <li>Visualizations: {pipeline_results.get('plots_file', 'N/A')}</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def export_complete_results(self, 
                               results_df: pd.DataFrame,
                               feature_importance: pd.DataFrame,
                               metrics: Dict[str, Any],
                               pipeline_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Export all results in a comprehensive package.
        This is the main export method for Phase 5.
        """
        logger.info("Exporting complete results package...")
        
        export_paths = {}
        
        try:
            # 1. Export risk scores
            scores_path = self.export_risk_scores(results_df)
            export_paths['risk_scores'] = scores_path
            logger.info(f"Risk scores exported to: {scores_path}")
            
            # 2. Export model metrics
            metrics_path = self.export_model_metrics(metrics)
            export_paths['model_metrics'] = metrics_path
            logger.info(f"Model metrics exported to: {metrics_path}")
            
            # 3. Export feature importance
            importance_path = self.export_feature_importance(feature_importance)
            export_paths['feature_importance'] = importance_path
            logger.info(f"Feature importance exported to: {importance_path}")
            
            # 4. Create risk distribution plot
            plot_path = self.create_risk_distribution_plot(results_df)
            export_paths['risk_distribution_plot'] = plot_path
            logger.info(f"Risk distribution plot saved to: {plot_path}")
            
            # 5. Create summary report
            report_path = self.create_summary_report(pipeline_results)
            export_paths['summary_report'] = report_path
            logger.info(f"Summary report created: {report_path}")
            
            logger.info(f"Complete results package exported successfully!")
            logger.info(f"Total files created: {len(export_paths)}")
            
            return export_paths
            
        except Exception as e:
            logger.error(f"Error exporting complete results: {str(e)}")
            return {}


class StreamlitDashboard:
    """
    Interactive Streamlit dashboard for the AI Payment Risk Scoring System.
    Provides visualization and analysis capabilities.
    """
    
    def __init__(self):
        """Initialize the dashboard."""
        self.setup_page_config()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="AI Payment Risk Scoring",
            page_icon="💳",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run_dashboard(self, results_df: pd.DataFrame = None, model_metrics: Dict = None):
        """Run the main dashboard interface."""
        st.title("🤖 AI Payment Risk Scoring Dashboard")
        st.markdown("---")
        
        # Sidebar
        self.create_sidebar()
        
        # Main content
        if results_df is not None:
            self.display_main_content(results_df, model_metrics)
        else:
            self.display_upload_interface()
    
    def create_sidebar(self):
        """Create the sidebar with navigation and controls."""
        st.sidebar.title("Navigation")
        
        sections = ["Overview", "Risk Analysis", "Model Performance", "Customer Details"]
        selected_section = st.sidebar.selectbox("Select Section", sections)
        
        return selected_section
    
    def display_upload_interface(self):
        """Display file upload interface."""
        st.header("📁 Data Upload")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Customer Data")
            uploaded_file = st.file_uploader(
                "Choose an Excel file", 
                type=['xlsx', 'xls'],
                help="Upload your customer payment data in Excel format"
            )
            
            if uploaded_file is not None:
                st.success("File uploaded successfully!")
                if st.button("Process Data"):
                    self.process_uploaded_file(uploaded_file)
        
        with col2:
            st.subheader("Sample Data")
            if st.button("Generate Sample Data"):
                self.generate_sample_data()
    
    def display_main_content(self, results_df: pd.DataFrame, model_metrics: Dict = None):
        """Display the main dashboard content."""
        # Overview section
        self.display_overview(results_df, model_metrics)
        
        # Risk analysis
        self.display_risk_analysis(results_df)
        
        # Model performance
        if model_metrics:
            self.display_model_performance(model_metrics)
        
        # Customer details
        self.display_customer_details(results_df)
    
    def display_overview(self, results_df: pd.DataFrame, model_metrics: Dict = None):
        """Display overview metrics."""
        st.header("📊 Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_customers = len(results_df)
            st.metric("Total Customers", total_customers)
        
        with col2:
            high_risk = len(results_df[results_df['risk_category'] == 'High Risk'])
            st.metric("High Risk Customers", high_risk)
        
        with col3:
            avg_risk_score = results_df['risk_score'].mean()
            st.metric("Average Risk Score", f"{avg_risk_score:.2f}")
        
        with col4:
            if model_metrics and 'accuracy' in model_metrics:
                accuracy = model_metrics['accuracy']
                st.metric("Model Accuracy", f"{accuracy:.3f}")
    
    def display_risk_analysis(self, results_df: pd.DataFrame):
        """Display risk analysis visualizations."""
        st.header("🎯 Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk score distribution
            fig_hist = px.histogram(
                results_df, 
                x='risk_score', 
                title='Risk Score Distribution',
                nbins=30,
                labels={'risk_score': 'Risk Score', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Risk category pie chart
            risk_counts = results_df['risk_category'].value_counts()
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title='Risk Category Distribution'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Risk score by features
        if 'age_group' in results_df.columns:
            fig_box = px.box(
                results_df,
                x='age_group',
                y='risk_score',
                title='Risk Score by Age Group'
            )
            st.plotly_chart(fig_box, use_container_width=True)
    
    def display_model_performance(self, model_metrics: Dict):
        """Display model performance metrics."""
        st.header("🎯 Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Classification Metrics")
            metrics_df = pd.DataFrame([{
                'Metric': k.replace('_', ' ').title(),
                'Value': f"{v:.4f}" if isinstance(v, float) else str(v)
            } for k, v in model_metrics.items() if k in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']])
            
            st.dataframe(metrics_df, use_container_width=True)
        
        with col2:
            st.subheader("Feature Importance")
            if 'feature_importance' in model_metrics:
                importance_data = model_metrics['feature_importance']
                if isinstance(importance_data, dict):
                    importance_df = pd.DataFrame(list(importance_data.items()), 
                                               columns=['Feature', 'Importance'])
                    importance_df = importance_df.sort_values('Importance', ascending=True)
                    
                    fig_bar = px.bar(
                        importance_df.tail(10),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 10 Feature Importance'
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
    
    def display_customer_details(self, results_df: pd.DataFrame):
        """Display detailed customer information."""
        st.header("👥 Customer Details")
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            risk_filter = st.selectbox(
                "Filter by Risk Category",
                ['All'] + list(results_df['risk_category'].unique())
            )
        
        with col2:
            score_range = st.slider(
                "Risk Score Range",
                min_value=float(results_df['risk_score'].min()),
                max_value=float(results_df['risk_score'].max()),
                value=(float(results_df['risk_score'].min()), float(results_df['risk_score'].max()))
            )
        
        # Apply filters
        filtered_df = results_df.copy()
        
        if risk_filter != 'All':
            filtered_df = filtered_df[filtered_df['risk_category'] == risk_filter]
        
        filtered_df = filtered_df[
            (filtered_df['risk_score'] >= score_range[0]) & 
            (filtered_df['risk_score'] <= score_range[1])
        ]
        
        # Display filtered data
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download option
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name=f"filtered_risk_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    def process_uploaded_file(self, uploaded_file):
        """Process uploaded file and run the pipeline."""
        st.info("Processing uploaded file...")
        # This would integrate with the main pipeline
        # For now, show a placeholder
        st.success("File processed successfully! Results will appear here.")
    
    def generate_sample_data(self):
        """Generate and display sample data."""
        st.info("Generating sample data...")
        # This would integrate with the data preparation module
        st.success("Sample data generated! Processing...")


# Legacy utility functions for backward compatibility
def load_excel_data(file_path):
    """Load data from Excel file."""
    return pd.read_excel(file_path)

def save_to_csv(dataframe, file_path):
    """Save dataframe to CSV file."""
    dataframe.to_csv(file_path, index=False)

def fill_missing_values(dataframe):
    """Fill missing values with median for numeric columns."""
    return dataframe.fillna(dataframe.median(numeric_only=True))

def get_top_features(shap_values, feature_names, top_n=3):
    """Get top contributing features from SHAP values."""
    top_features = []
    for i in range(len(shap_values)):
        if hasattr(shap_values[i], 'values'):
            values = shap_values[i].values
        else:
            values = shap_values[i]
        
        top_idx = np.argsort(np.abs(values))[-top_n:][::-1]
        top_feats = [feature_names[j] for j in top_idx]
        top_features.append(", ".join(top_feats))
    return top_features

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('payment_risk_scoring.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_directories(directories: List[str]):
    """Create directories if they don't exist."""
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def load_model(model_path: str):
    """Load a saved model."""
    return joblib.load(model_path)

def save_model(model, model_path: str):
    """Save a model to disk."""
    joblib.dump(model, model_path)
"""
Phase 1: Data Preparation Module
This module handles loading, cleaning, and preparing the Excel data for ML training
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG, DATA_PATH, PROCESSED_DATA_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreparator:
    """Class to handle data preparation tasks"""
    
    def __init__(self, excel_file_path: str = None):
        self.excel_file_path = excel_file_path
        self.raw_data = None
        self.processed_data = None
        
    def load_excel_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Step 1.1: Load and inspect the Excel data
        """
        if file_path:
            self.excel_file_path = file_path
            
        if not self.excel_file_path:
            # Look for Excel files in the data directory
            excel_files = [f for f in os.listdir(DATA_PATH) if f.endswith(('.xlsx', '.xls'))]
            if excel_files:
                self.excel_file_path = os.path.join(DATA_PATH, excel_files[0])
                logger.info(f"Found Excel file: {excel_files[0]}")
            else:
                # Create sample data if no Excel file exists
                logger.warning("No Excel file found. Creating sample data for demonstration.")
                return self.create_sample_data()
        
        try:
            logger.info(f"Loading Excel file: {self.excel_file_path}")
            self.raw_data = pd.read_excel(self.excel_file_path)
            
            logger.info(f"Data loaded successfully. Shape: {self.raw_data.shape}")
            logger.info(f"Columns: {list(self.raw_data.columns)}")
            
            return self.raw_data
            
        except FileNotFoundError:
            logger.error(f"File not found: {self.excel_file_path}")
            logger.info("Creating sample data for demonstration.")
            return self.create_sample_data()
        except Exception as e:
            logger.error(f"Error loading Excel file: {str(e)}")
            return self.create_sample_data()
    
    def create_sample_data(self) -> pd.DataFrame:
        """Create sample data for demonstration purposes"""
        logger.info("Creating sample customer payment data...")
        
        np.random.seed(42)
        n_customers = 50
        n_calls_per_customer = np.random.randint(1, 5, n_customers)
        
        data = []
        for customer_id in range(1, n_customers + 1):
            for call in range(n_calls_per_customer[customer_id - 1]):
                # Create realistic customer data
                payment_history = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% payment issues
                
                record = {
                    'レコード番号': customer_id,
                    'コンタクト_id': f"CONTACT_{customer_id}_{call + 1}",
                    '開始タイムスタンプjst': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 365)),
                    'エージェント': f"Agent_{np.random.randint(1, 10)}",
                    '営業担当者': f"Sales_{np.random.randint(1, 5)}",
                    '年収': np.random.normal(500, 150) * 10000,  # Annual income
                    '勤続年数': np.random.exponential(5),  # Years at job
                    '年齢': np.random.normal(40, 12),  # Age
                    '借入額': np.random.exponential(200) * 10000,  # Loan amount
                    '月収': np.random.normal(42, 12) * 10000,  # Monthly income
                    '家族構成': np.random.randint(1, 6),  # Family size
                    '住宅ローン': np.random.choice([0, 1], p=[0.4, 0.6]),  # Has mortgage
                    'クレジットスコア': np.random.normal(650, 100),  # Credit score
                    '過去延滞回数': np.random.poisson(payment_history * 2),  # Past delinquencies
                    '通話時間秒': np.random.exponential(300),  # Call duration in seconds
                    '通話回数': call + 1,  # Number of calls
                    '未払FLAG': payment_history  # Target variable
                }
                data.append(record)
        
        self.raw_data = pd.DataFrame(data)
        
        # Save sample data
        sample_file_path = os.path.join(DATA_PATH, "sample_customer_data.xlsx")
        self.raw_data.to_excel(sample_file_path, index=False)
        logger.info(f"Sample data saved to: {sample_file_path}")
        
        return self.raw_data
    
    def aggregate_customer_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Step 1.2: Group by customer ID and aggregate multiple calls per customer
        """
        if df is None:
            df = self.raw_data
        
        if df is None:
            raise ValueError("No data available. Please load data first.")
        
        logger.info("Aggregating data by customer...")
        
        # Separate numeric and non-numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and ID columns from numeric aggregation
        target_col = DATA_CONFIG['target_column']
        customer_id_col = DATA_CONFIG['customer_id_column']
        
        if target_col in numeric_columns:
            numeric_columns.remove(target_col)
        if customer_id_col in numeric_columns:
            numeric_columns.remove(customer_id_col)
        
        # Aggregate numeric columns (mean)
        customer_df = df.groupby(customer_id_col)[numeric_columns].mean().reset_index()
        
        # Keep target variable (first occurrence per customer)
        customer_df[target_col] = df.groupby(customer_id_col)[target_col].first().values
        
        logger.info(f"Data aggregated. Shape: {customer_df.shape}")
        return customer_df
    
    def clean_and_engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 1.3 & 1.4: Feature engineering and cleaning
        """
        logger.info("Cleaning and engineering features...")
        
        # Drop irrelevant columns
        drop_cols = DATA_CONFIG['drop_columns']
        df_clean = df.drop(columns=[col for col in drop_cols if col in df.columns])
        
        # Handle missing values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_columns] = df_clean[numeric_columns].fillna(df_clean[numeric_columns].median())
        
        # Feature engineering
        if '年収' in df_clean.columns and '借入額' in df_clean.columns:
            # Debt-to-income ratio
            df_clean['債務収入比率'] = df_clean['借入額'] / (df_clean['年収'] + 1)
        
        if '月収' in df_clean.columns and '借入額' in df_clean.columns:
            # Monthly debt burden
            df_clean['月次債務負担'] = df_clean['借入額'] / (df_clean['月収'] * 12 + 1)
        
        if '年齢' in df_clean.columns:
            # Age groups
            df_clean['年齢グループ'] = pd.cut(df_clean['年齢'], 
                                       bins=[0, 30, 40, 50, 60, 100], 
                                       labels=[1, 2, 3, 4, 5])
            df_clean['年齢グループ'] = df_clean['年齢グループ'].astype(float)
        
        if 'クレジットスコア' in df_clean.columns:
            # Credit score categories
            df_clean['クレジット区分'] = pd.cut(df_clean['クレジットスコア'],
                                        bins=[0, 550, 650, 750, 850],
                                        labels=[1, 2, 3, 4])
            df_clean['クレジット区分'] = df_clean['クレジット区分'].astype(float)
        
        # Handle any remaining missing values
        df_clean = df_clean.fillna(0)
        
        logger.info(f"Feature engineering completed. Final shape: {df_clean.shape}")
        logger.info(f"Final columns: {list(df_clean.columns)}")
        
        self.processed_data = df_clean
        return df_clean
    
    def prepare_ml_data(self, df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features (X) and target (y) for machine learning
        """
        if df is None:
            df = self.processed_data
        
        if df is None:
            raise ValueError("No processed data available. Please run data preparation first.")
        
        # Separate features and target
        target_col = DATA_CONFIG['target_column']
        customer_id_col = DATA_CONFIG['customer_id_column']
        
        # Features (X) - all columns except target and customer ID
        feature_cols = [col for col in df.columns if col not in [target_col, customer_id_col]]
        X = df[feature_cols]
        
        # Target (y) - encode Japanese text to binary
        y = df[target_col]
        
        # Encode Japanese target values to binary (0/1)
        if y.dtype == 'object':
            logger.info("Encoding Japanese target values to binary...")
            # Map Japanese values: 未払 (unpaid) -> 1, 支払済 (paid) -> 0
            target_mapping = {'未払': 1, '支払済': 0}
            y = y.map(target_mapping)
            
            # Check if mapping was successful
            if y.isnull().any():
                unique_values = df[target_col].unique()
                logger.warning(f"Found unmapped target values: {unique_values}")
                # Fallback mapping for any unexpected values
                y = y.fillna(0)
            
            logger.info(f"Target encoding completed: 未払=1 (high risk), 支払済=0 (low risk)")
        
        logger.info(f"ML data prepared. Features: {X.shape[1]}, Samples: {X.shape[0]}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def save_processed_data(self, df: pd.DataFrame = None, filename: str = "processed_customer_data.csv"):
        """Save processed data to file"""
        if df is None:
            df = self.processed_data
        
        if df is None:
            raise ValueError("No processed data to save.")
        
        file_path = os.path.join(PROCESSED_DATA_PATH, filename)
        df.to_csv(file_path, index=False)
        logger.info(f"Processed data saved to: {file_path}")
        return file_path
    
    def run_full_preparation(self, excel_file_path: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Run the complete data preparation pipeline
        """
        logger.info("Starting full data preparation pipeline...")
        
        # Step 1.1: Load data (try real data first, fall back to sample)
        try:
            if DATA_CONFIG.get('use_real_data', False) or excel_file_path:
                logger.info("Attempting to load real data...")
                raw_df = self.load_real_data(excel_file_path)
            else:
                logger.info("Loading sample data...")
                raw_df = self.load_excel_data(excel_file_path)
            
            # Check if this looks like real call center data
            real_data_indicators = [
                'agent_sentiment_score', 'customer_sentiment_score',
                'agent_talktime', 'customer_talktime', 'total_conversation_duration'
            ]
            
            has_real_indicators = any(col in raw_df.columns for col in real_data_indicators)
            
            if has_real_indicators:
                logger.info("Detected real call center data structure")
                
                # Step 1.2: Clean real data
                cleaned_df = self.clean_real_data(raw_df)
                
                # Step 1.3: Engineer call center specific features
                featured_df = self.engineer_features_real_data(cleaned_df)
                
                # Step 1.4: Aggregate by customer
                processed_df = self.aggregate_real_data(featured_df)
                
            else:
                logger.info("Using standard data processing pipeline")
                
                # Step 1.2: Aggregate by customer
                customer_df = self.aggregate_customer_data(raw_df)
                
                # Step 1.3 & 1.4: Clean and engineer features
                processed_df = self.clean_and_engineer_features(customer_df)
            
        except Exception as e:
            logger.warning(f"Real data loading failed: {str(e)}")
            logger.info("Falling back to sample data...")
            raw_df = self.load_excel_data()
            customer_df = self.aggregate_customer_data(raw_df)
            processed_df = self.clean_and_engineer_features(customer_df)
        
        # Save processed data
        self.save_processed_data(processed_df)
        
        # Prepare ML data
        X, y = self.prepare_ml_data(processed_df)
        
        logger.info("Data preparation pipeline completed successfully!")
        return X, y
    
    def load_real_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load and validate real customer data with your specific columns
        """
        logger.info("Loading real customer data...")
        
        # Use provided path or look for Excel files
        if file_path is None:
            excel_files = [f for f in os.listdir(DATA_PATH) if f.endswith(('.xlsx', '.xls'))]
            if excel_files:
                file_path = os.path.join(DATA_PATH, excel_files[0])
            else:
                raise FileNotFoundError("No Excel file found in data directory")
        
        try:
            # Load the Excel file
            logger.info(f"Loading real data from: {file_path}")
            self.raw_data = pd.read_excel(file_path)
            
            logger.info(f"Real data loaded successfully. Shape: {self.raw_data.shape}")
            logger.info(f"Columns found: {len(self.raw_data.columns)}")
            
            # Validate required columns
            required_columns = [DATA_CONFIG['target_column'], DATA_CONFIG['customer_id_column']]
            missing_columns = [col for col in required_columns if col not in self.raw_data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Log target variable distribution
            target_col = DATA_CONFIG['target_column']
            if target_col in self.raw_data.columns:
                target_dist = self.raw_data[target_col].value_counts()
                logger.info(f"Target variable distribution:\n{target_dist}")
            
            # Log data quality info
            missing_data = self.raw_data.isnull().sum()
            if missing_data.sum() > 0:
                logger.warning(f"Missing data found in {missing_data[missing_data > 0].shape[0]} columns")
                for col, count in missing_data[missing_data > 0].items():
                    logger.warning(f"  - {col}: {count} missing values")
            
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error loading real data: {str(e)}")
            raise
    
    def clean_real_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare real data with your specific column structure
        """
        logger.info("Cleaning real customer data...")
        
        # Make a copy to avoid modifying original data
        df_clean = df.copy()
        
        # Drop configured columns that should be excluded
        drop_cols = DATA_CONFIG.get('drop_columns', [])
        existing_drop_cols = [col for col in drop_cols if col in df_clean.columns]
        if existing_drop_cols:
            df_clean = df_clean.drop(columns=existing_drop_cols)
            logger.info(f"Dropped columns: {existing_drop_cols}")
        
        # Handle date columns if they exist
        date_columns = DATA_CONFIG.get('date_columns', [])
        for date_col in date_columns:
            if date_col in df_clean.columns:
                try:
                    df_clean[date_col] = pd.to_datetime(df_clean[date_col])
                    logger.info(f"Converted {date_col} to datetime")
                except:
                    logger.warning(f"Could not convert {date_col} to datetime")
        
        # Handle missing values for numeric columns
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                logger.info(f"Filled missing values in {col} with median: {median_val:.2f}")
        
        # Handle missing values for categorical columns
        categorical_columns = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                df_clean[col] = df_clean[col].fillna(mode_val)
                logger.info(f"Filled missing values in {col} with mode: {mode_val}")
        
        logger.info(f"Data cleaning completed. Shape: {df_clean.shape}")
        return df_clean
    
    def engineer_features_real_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features specific to your call center data
        """
        logger.info("Engineering features for call center data...")
        
        df_features = df.copy()
        
        # Agent vs Customer interaction ratios
        if 'agent_talktime' in df_features.columns and 'customer_talktime' in df_features.columns:
            df_features['agent_customer_talktime_ratio'] = (
                df_features['agent_talktime'] / (df_features['customer_talktime'] + 1)
            )
            logger.info("Created agent-customer talktime ratio")
        
        # Sentiment analysis features
        if 'agent_sentiment_score' in df_features.columns and 'customer_sentiment_score' in df_features.columns:
            df_features['sentiment_difference'] = (
                df_features['agent_sentiment_score'] - df_features['customer_sentiment_score']
            )
            df_features['avg_sentiment'] = (
                df_features['agent_sentiment_score'] + df_features['customer_sentiment_score']
            ) / 2
            logger.info("Created sentiment-based features")
        
        # Communication efficiency features
        if 'agent_total_sentence' in df_features.columns and 'customer_total_sentence' in df_features.columns:
            df_features['total_sentences'] = (
                df_features['agent_total_sentence'] + df_features['customer_total_sentence']
            )
            df_features['agent_sentence_ratio'] = (
                df_features['agent_total_sentence'] / (df_features['total_sentences'] + 1)
            )
            logger.info("Created communication efficiency features")
        
        # Conversation quality features
        if 'total_conversation_duration' in df_features.columns and 'total_talktime' in df_features.columns:
            df_features['silence_ratio'] = 1 - (
                df_features['total_talktime'] / (df_features['total_conversation_duration'] + 1)
            )
            logger.info("Created conversation quality features")
        
        # Agent performance features
        agent_positive_cols = [col for col in df_features.columns if 'agent_positive' in col]
        agent_negative_cols = [col for col in df_features.columns if 'agent_negative' in col]
        
        if agent_positive_cols and agent_negative_cols:
            df_features['agent_positivity_ratio'] = (
                df_features[agent_positive_cols[0]] / 
                (df_features[agent_positive_cols[0]] + df_features[agent_negative_cols[0]] + 1)
            )
            logger.info("Created agent positivity ratio")
        
        # Customer engagement features  
        customer_positive_cols = [col for col in df_features.columns if 'customer_positive' in col]
        customer_negative_cols = [col for col in df_features.columns if 'customer_negative' in col]
        
        if customer_positive_cols and customer_negative_cols:
            df_features['customer_positivity_ratio'] = (
                df_features[customer_positive_cols[0]] / 
                (df_features[customer_positive_cols[0]] + df_features[customer_negative_cols[0]] + 1)
            )
            logger.info("Created customer positivity ratio")
        
        # Time-based features (if date columns exist)
        if '電話日-成約日' in df_features.columns:
            # This seems to be a time difference feature already
            logger.info("Found time difference feature: 電話日-成約日")
        
        logger.info(f"Feature engineering completed. New shape: {df_features.shape}")
        logger.info(f"New features created: {df_features.shape[1] - df.shape[1]}")
        
        return df_features

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and engineer features for the customer data.
        This is a wrapper around existing functionality to maintain compatibility.
        
        Args:
            df (pd.DataFrame): The dataframe to process
            
        Returns:
            pd.DataFrame: The processed dataframe with engineered features
        """
        logger.info("Engineering features for the dataset...")
        
        # Check if this looks like real call center data
        real_data_indicators = [
            'agent_sentiment_score', 'customer_sentiment_score',
            'agent_talktime', 'customer_talktime', 'total_conversation_duration'
        ]
        
        has_real_indicators = any(col in df.columns for col in real_data_indicators)
        
        if has_real_indicators:
            logger.info("Detected real call center data structure")
            # Use the real data feature engineering method
            processed_df = self.engineer_features_real_data(df)
        else:
            logger.info("Using standard data processing pipeline")
            # Use the standard feature cleaning and engineering method
            processed_df = self.clean_and_engineer_features(df)
            
        logger.info(f"Feature engineering completed. Shape: {processed_df.shape}")
        return processed_df
        
    def aggregate_real_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate multiple calls per customer for real data
        """
        logger.info("Aggregating real data by customer...")
        
        customer_id_col = DATA_CONFIG['customer_id_column']
        target_col = DATA_CONFIG['target_column']
        
        # Separate numeric and categorical columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove ID and target from numeric aggregation
        agg_numeric_cols = [col for col in numeric_columns if col not in [customer_id_col, target_col]]
        
        # Aggregate numeric columns (mean for most, sum for counts)
        agg_dict = {}
        for col in agg_numeric_cols:
            if 'sentence' in col.lower() or 'count' in col.lower():
                agg_dict[col] = 'sum'  # Sum for count-based features
            else:
                agg_dict[col] = 'mean'  # Mean for other numeric features
        
        # Perform aggregation
        customer_df = df.groupby(customer_id_col).agg(agg_dict).reset_index()
        
        # Keep target variable (take the mode/most common value per customer)
        customer_df[target_col] = df.groupby(customer_id_col)[target_col].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]).values
        
        # Add call count feature
        customer_df['call_count'] = df.groupby(customer_id_col).size().values
        
        logger.info(f"Data aggregated by customer. Shape: {customer_df.shape}")
        logger.info(f"Unique customers: {customer_df[customer_id_col].nunique()}")
        
        return customer_df


# Legacy functions for backward compatibility
def load_data(file_path):
    """Load Excel data from the specified file path."""
    df = pd.read_excel(file_path)
    return df

def inspect_data(df):
    """Inspect the data by returning the first few rows and columns."""
    print(df.head())
    print(df.columns)

def group_by_customer(df):
    """Group the data by customer ID and aggregate numeric fields."""
    customer_df = df.groupby('レコード番号').mean(numeric_only=True).reset_index()
    customer_df['未払FLAG'] = df.groupby('レコード番号')['未払FLAG'].first().values
    return customer_df

def select_features(customer_df):
    """Select useful features and drop irrelevant columns."""
    drop_cols = ['開始タイムスタンプjst', 'コンタクト_id', 'エージェント', '営業担当者']
    customer_df = customer_df.drop(columns=[col for col in drop_cols if col in customer_df.columns])
    return customer_df

def handle_missing_values(customer_df):
    """Handle missing values by filling them with the median."""
    customer_df = customer_df.fillna(customer_df.median(numeric_only=True))
    return customer_df


def main():
    """Main function for testing data preparation"""
    preparator = DataPreparator()
    X, y = preparator.run_full_preparation()
    
    print("\n=== Data Preparation Summary ===")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    print(f"Target distribution:\n{y.value_counts()}")


if __name__ == "__main__":
    main()
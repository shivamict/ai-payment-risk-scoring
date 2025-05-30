# Configuration settings for the AI Payment Risk Scoring project
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUTS_PATH = os.path.join(BASE_DIR, 'outputs')
MODELS_PATH = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create directories if they don't exist
try:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
except FileExistsError:
    # Handle case where directories already exist
    pass

# Data configuration
DATA_CONFIG = {
    'target_column': '未払FLAG',
    'customer_id_column': 'レコード番号',
    'date_columns': [
        '成約日date',
        '開始タイムスタンプjst'
    ],
    'drop_columns': [
        '開始タイムスタンプjst', 
        'コンタクト_id', 
        'エージェント', 
        '営業担当者',
        'content_list'  # Text data - might be too complex for initial model
    ],
    'feature_columns': [
        'agent_loudness_mean',
        'agent_negative_sentence',
        'agent_neutral_sentence', 
        'agent_positive_sentence',
        'agent_sentiment_score',
        'agent_talktime',
        'agent_talktime通話時間',
        'agent_total_sentence',
        'customer_loudness_mean',
        'customer_negative_sentence',
        'customer_neutral_sentence',
        'customer_positive_sentence',
        'customer_sentiment_score',
        'customer_talktime',
        'customer_talktime通話時間',
        'customer_total_sentence',
        'total_conversation_duration',
        'total_conversation_duration合計',
        'total_talktime',
        'total_talktime通話時間',
        '電話日-成約日'
    ],
    'test_size': 0.2,
    'random_state': 42,
    'use_real_data': True,
    'real_data_path': 'data/raw/real_customer_data.xlsx'  # You'll need to place your file here
}

# Model parameters
XGBOOST_PARAMS = {
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'learning_rate': 0.1,
    'max_depth': 6,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# Risk scoring configuration
SCORING_CONFIG = {
    'score_min': 0,
    'score_max': 100,
    'risk_thresholds': {
        'low': 70,
        'medium': 40,
        'high': 0
    }
}

# SHAP parameters
SHAP_EXPLAINER_PARAMS = {
    'model': None,  # To be set after model training
    'data': None    # To be set after data preparation
}

# Configuration aliases for test compatibility
MODEL_PARAMS = XGBOOST_PARAMS
RISK_THRESHOLDS = SCORING_CONFIG['risk_thresholds']

# Dashboard configuration
DASHBOARD_CONFIG = {
    'port': 8501,
    'host': 'localhost',
    'title': 'AI Payment Risk Scoring Dashboard'
}
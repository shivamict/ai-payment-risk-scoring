# Fix the DataPreparator class if needed
import os
import sys
from pathlib import Path

# Add the missing method to DataPreparator
data_prep_path = Path('src/data_preparation.py')
if data_prep_path.exists():
    with open(data_prep_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if the method already exists
    if 'def engineer_features_japanese' not in content:
        # Add the method to the class
        insert_pos = content.find('class DataPreparator:') + len('class DataPreparator:')
        
        new_method = '''
    def engineer_features_japanese(self, df):
        """
        日本語の顧客データに対する特徴量エンジニアリング
        """
        print("🔧 日本語データ用の特徴エンジニアリングを実行中...")
        
        # 基本的な特徴量の作成
        engineered_df = df.copy()
        
        # エージェントと顧客の通話時間比率
        if 'agent_talktime' in df.columns and 'customer_talktime' in df.columns:
            engineered_df['agent_customer_talktime_ratio'] = df['agent_talktime'] / df['customer_talktime'].replace(0, 0.001)
        
        # 感情スコアの差異
        if 'agent_sentiment_score' in df.columns and 'customer_sentiment_score' in df.columns:
            engineered_df['sentiment_difference'] = df['agent_sentiment_score'] - df['customer_sentiment_score']
            engineered_df['avg_sentiment'] = (df['agent_sentiment_score'] + df['customer_sentiment_score']) / 2
        
        # 文の合計と比率
        if 'agent_total_sentence' in df.columns and 'customer_total_sentence' in df.columns:
            engineered_df['total_sentences'] = df['agent_total_sentence'] + df['customer_total_sentence']
            engineered_df['agent_sentence_ratio'] = df['agent_total_sentence'] / engineered_df['total_sentences'].replace(0, 1)
        
        # 無音時間の比率
        if 'total_conversation_duration' in df.columns and 'total_talktime' in df.columns:
            engineered_df['silence_ratio'] = 1 - (df['total_talktime'] / df['total_conversation_duration'].replace(0, 1))
        
        # ポジティブな文の比率
        if 'agent_positive_sentence' in df.columns and 'agent_total_sentence' in df.columns:
            engineered_df['agent_positivity_ratio'] = df['agent_positive_sentence'] / df['agent_total_sentence'].replace(0, 1)
        
        if 'customer_positive_sentence' in df.columns and 'customer_total_sentence' in df.columns:
            engineered_df['customer_positivity_ratio'] = df['customer_positive_sentence'] / df['customer_total_sentence'].replace(0, 1)
        
        print(f" 特徴エンジニアリング完了。生成された特徴量: {len(engineered_df.columns)}")
        return engineered_df
'''
        
        new_content = content[:insert_pos] + new_method + content[insert_pos:]
        
        with open(data_prep_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(" DataPreparator クラスに engineer_features_japanese メソッドを追加しました")

# Fix the optimize_model.py file
optimize_model_path = Path('optimize_model.py')
if optimize_model_path.exists():
    with open(optimize_model_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the load_and_prepare_data method
    start_marker = 'def load_and_prepare_data(self):'
    end_marker = 'return X, y'
    
    start_pos = content.find(start_marker)
    if start_pos > -1:
        # Find the end of the method
        search_from = start_pos + len(start_marker)
        end_pos = content.find(end_marker, search_from) + len(end_marker)
        
        new_method = '''def load_and_prepare_data(self):
        """Load and prepare Japanese data for optimization."""
        logger.info(" 日本語データの読み込みと準備中...")
        
        # Check if processed data exists
        processed_data_path = Path("outputs/processed_japanese_data.csv")
        
        if processed_data_path.exists():
            logger.info(f" 処理済みデータを読み込み中: {processed_data_path}")
            data = pd.read_csv(processed_data_path)
            
            # Check if the target column exists
            if '未払FLAG' not in data.columns:
                logger.error(" エラー: '未払FLAG'列が見つかりません。")
                raise ValueError("Target column '未払FLAG' not found in processed data")
            
            # Prepare features and target
            y = data['未払FLAG']
            X = data.drop(['未払FLAG', 'レコード番号'], axis=1, errors='ignore')
            
            # Convert all columns to numeric, errors to NaN
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            
            # Fill any missing values
            X = X.fillna(X.mean())
            
            logger.info(f" データセットの形状: {X.shape}")
            logger.info(f" ターゲット分布: {dict(pd.Series(y).value_counts())}")
            
            return X, y
        else:
            # Try to find raw data
            excel_files = list(Path("data/raw").glob("*.xlsx"))
            
            if not excel_files:
                logger.error(" データが見つかりません。処理済みCSVまたは生のExcelファイルが必要です。")
                raise FileNotFoundError("No data found. Please upload data first.")
            
            # Process the first Excel file found
            logger.info(f" Excelファイルを処理中: {excel_files[0]}")
            
            # Read Excel file
            data = pd.read_excel(excel_files[0])
            
            # Check if the target column exists
            if '未払FLAG' not in data.columns:
                logger.error(" エラー: '未払FLAG'列が見つかりません。")
                raise ValueError("Target column '未払FLAG' not found in Excel data")
            
            # Process data using DataPreparator
            if hasattr(self.data_prep, 'engineer_features_japanese'):
                processed_data = self.data_prep.engineer_features_japanese(data)
            else:
                processed_data = self.data_prep.engineer_features_real_data(data)
            
            # Save processed data
            processed_data.to_csv("outputs/processed_japanese_data.csv", index=False)
            
            # Prepare features and target
            y = processed_data['未払FLAG']
            X = processed_data.drop(['未払FLAG', 'レコード番号'], axis=1, errors='ignore')
            
            # Convert all columns to numeric, errors to NaN
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            
            # Fill any missing values
            X = X.fillna(X.mean())
            
            logger.info(f" データセットの形状: {X.shape}")
            logger.info(f" ターゲット分布: {dict(pd.Series(y).value_counts())}")
            
            return X, y'''
        
        new_content = content[:start_pos] + new_method + content[end_pos:]
        
        with open(optimize_model_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(" optimize_model.py ファイルを修正しました")

print(" 修正が完了しました。optimize_model.py を実行してください。")

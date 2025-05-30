import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preparation import DataPreparator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """最適化されたAIリスクスコアリングモデルを作成するクラス"""
    
    def __init__(self):
        """モデルオプティマイザーの初期化"""
        self.data_prep = DataPreparator()
        self.best_model = None
        self.feature_importances = None
        self.best_params = None
        self.baseline_auc = None
        self.optimized_auc = None
        self.selected_features = None
    
    def load_and_prepare_data(self):
        """Load and prepare Japanese data for optimization."""
        logger.info("🔄 日本語データの読み込みと準備中...")
        
        # Check if processed data exists
        processed_data_path = Path("outputs/processed_japanese_data.csv")
        
        if processed_data_path.exists():
            logger.info(f"✅ 処理済みデータを読み込み中: {processed_data_path}")
            data = pd.read_csv(processed_data_path)
            
            # Check if the target column exists
            if '未払FLAG' not in data.columns:
                logger.error("❌ エラー: '未払FLAG'列が見つかりません。")
                raise ValueError("Target column '未払FLAG' not found in processed data")
            
            # Prepare features and target
            y_raw = data['未払FLAG']
            
            # Convert string labels to numeric (0 for '支払済', 1 for '未払')
            y = y_raw.map(lambda x: 1 if x == '未払' else 0)
            logger.info(f"✅ ラベルを数値に変換: '支払済' → 0, '未払' → 1")
            
            X = data.drop(['未払FLAG', 'レコード番号'], axis=1, errors='ignore')
            
            # Convert all columns to numeric, errors to NaN
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            
            # Fill any missing values
            X = X.fillna(X.mean())
            
            logger.info(f"📊 データセットの形状: {X.shape}")
            logger.info(f"🎯 ターゲット分布: {dict(pd.Series(y).value_counts())}")
            
            return X, y
        
        else:
            # Try to find raw data
            excel_files = list(Path("data/raw").glob("*.xlsx"))
            
            if not excel_files:
                logger.error("❌ データが見つかりません。処理済みCSVまたは生のExcelファイルが必要です。")
                raise FileNotFoundError("No data found. Please upload data first.")
            
            # Process the first Excel file found
            logger.info(f"📊 Excelファイルを処理中: {excel_files[0]}")
            
            # Read Excel file
            data = pd.read_excel(excel_files[0])
            
            # Check if the target column exists
            if '未払FLAG' not in data.columns:
                logger.error("❌ エラー: '未払FLAG'列が見つかりません。")
                raise ValueError("Target column '未払FLAG' not found in Excel data")
            
            # Process data using DataPreparator
            if hasattr(self.data_prep, 'engineer_features_japanese'):
                processed_data = self.data_prep.engineer_features_japanese(data)
            else:
                processed_data = self.data_prep.engineer_features_real_data(data)
            
            # Save processed data
            processed_data.to_csv("outputs/processed_japanese_data.csv", index=False)
            
            # Prepare features and target
            y_raw = processed_data['未払FLAG']
            
            # Convert string labels to numeric (0 for '支払済', 1 for '未払')
            y = y_raw.map(lambda x: 1 if x == '未払' else 0)
            logger.info(f"✅ ラベルを数値に変換: '支払済' → 0, '未払' → 1")
            
            X = processed_data.drop(['未払FLAG', 'レコード番号'], axis=1, errors='ignore')
            
            # Convert all columns to numeric, errors to NaN
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            
            # Fill any missing values
            X = X.fillna(X.mean())
            
            logger.info(f"📊 データセットの形状: {X.shape}")
            logger.info(f"🎯 ターゲット分布: {dict(pd.Series(y).value_counts())}")
            
            return X, y
    
    def evaluate_baseline_model(self, X, y):
        """ベースラインモデルの評価"""
        logger.info(" ベースラインモデルの評価...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train a simple Random Forest as baseline
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Evaluate
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f" ベースラインAUC: {auc:.4f}")
        self.baseline_auc = auc
        
        return rf, auc
    
    def select_features(self, X, y):
        """特徴量選択"""
        logger.info(" 最適な特徴量の選択...")
        
        # Train a model for feature selection
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Select important features
        selector = SelectFromModel(rf, threshold='median')
        selector.fit(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()]
        
        # Get importance scores
        importances = rf.feature_importances_
        feature_importances = pd.DataFrame({
            '特徴量': X.columns,
            '重要度': importances
        }).sort_values('重要度', ascending=False)
        
        logger.info(f" 選択された特徴量: {len(selected_features)}/{X.shape[1]}")
        logger.info(f" トップ5特徴量: {', '.join(feature_importances['特徴量'].head(5))}")
        
        self.feature_importances = feature_importances
        self.selected_features = selected_features
        
        return X[selected_features], selected_features
    
    def handle_imbalance(self, X, y):
        """不均衡データの処理"""
        logger.info(" 不均衡データの処理...")
        
        # Check class distribution
        class_counts = pd.Series(y).value_counts()
        logger.info(f" クラス分布: {class_counts.to_dict()}")
        
        # Apply SMOTE if imbalanced
        min_samples = class_counts.min()
        if min_samples < 10:
            logger.warning(f" サンプルが少なすぎるため、SMOTEをスキップします ({min_samples})")
            return X, y
        
        try:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            logger.info(f" SMOTE適用後: {pd.Series(y_resampled).value_counts().to_dict()}")
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f" SMOTE適用エラー: {e}")
            return X, y
    
    def optimize_xgboost(self, X, y):
        """XGBoostモデルの最適化"""
        logger.info(" XGBoostモデルのパラメータ最適化...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'min_child_weight': [1, 3]
        }
        
        # Use smaller grid for small datasets
        if X.shape[0] < 100:
            logger.info(" 小規模データセットのため、簡易グリッドを使用")
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.1]
            }
        
        # Grid search with cross-validation
        xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
        )
        
        try:
            grid_search.fit(X_train, y_train)
            
            # Get best parameters
            best_params = grid_search.best_params_
            logger.info(f" 最適パラメータ: {best_params}")
            
            # Train with best parameters
            best_model = xgb.XGBClassifier(
                objective='binary:logistic', 
                random_state=42,
                **best_params
            )
            best_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            
            logger.info(f" 最適化後AUC: {auc:.4f}")
            
            self.best_model = best_model
            self.best_params = best_params
            self.optimized_auc = auc
            
            return best_model, best_params, auc
            
        except Exception as e:
            logger.error(f" XGBoost最適化エラー: {e}")
            
            # Fallback to simpler model
            logger.info(" 簡易XGBoostモデルにフォールバック")
            simple_model = xgb.XGBClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
            )
            simple_model.fit(X_train, y_train)
            
            y_pred_proba = simple_model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            
            logger.info(f" フォールバックモデルAUC: {auc:.4f}")
            
            self.best_model = simple_model
            self.best_params = {
                'n_estimators': 50, 
                'max_depth': 3, 
                'learning_rate': 0.1
            }
            self.optimized_auc = auc
            
            return simple_model, self.best_params, auc
    
    def compare_algorithms(self, X, y):
        """複数のアルゴリズムの比較"""
        logger.info(" 複数のアルゴリズムを比較...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models to compare
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
                
                results[name] = auc
                logger.info(f" {name} AUC: {auc:.4f}")
                
            except Exception as e:
                logger.error(f" {name}モデルエラー: {e}")
                results[name] = 0
        
        # Find best model
        best_algo = max(results, key=results.get)
        logger.info(f" 最高性能アルゴリズム: {best_algo} (AUC: {results[best_algo]:.4f})")
        
        return results
    
    def generate_model_report(self, X, y):
        """モデルレポートの生成"""
        logger.info(" モデルレポートの生成...")
        
        if self.best_model is None:
            logger.error(" 最適化されたモデルがありません")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model on this split
        self.best_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Log results
        logger.info(f" テストセットAUC: {auc:.4f}")
        logger.info(f" 混同行列:\n{cm}")
        logger.info(f" 分類レポート:\n{report}")
        
        # Save results
        results = {
            'auc': auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'feature_importances': self.feature_importances.to_dict() if self.feature_importances is not None else None,
            'best_params': self.best_params,
            'baseline_auc': self.baseline_auc,
            'optimized_auc': self.optimized_auc,
            'improvement': (self.optimized_auc - self.baseline_auc) if self.baseline_auc and self.optimized_auc else None
        }
        
        # Create output directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        
        # Save report as JSON
        report_path = os.path.join('outputs', 'model_report.json')
        with open(report_path, 'w') as f:
            import json
            json.dump(results, f, indent=4)
        
        logger.info(f" モデルレポート保存先: {report_path}")
        
        # Create visualizations
        self.create_visualizations(X, y, X_test, y_test, y_pred_proba)
        
        return results
    
    def create_visualizations(self, X, y, X_test, y_test, y_pred_proba):
        """可視化の作成"""
        logger.info(" モデル性能の可視化...")
        
        # Create output directory for plots
        plots_dir = os.path.join('outputs', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        try:
            # 1. Feature Importance
            if self.feature_importances is not None:
                plt.figure(figsize=(10, 6))
                top_features = self.feature_importances.head(10)
                sns.barplot(x='重要度', y='特徴量', data=top_features)
                plt.title('特徴量重要度 (トップ10)')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
                plt.close()
            
            # 2. ROC Curve
            from sklearn.metrics import roc_curve
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'AUC = {self.optimized_auc:.4f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('偽陽性率')
            plt.ylabel('真陽性率')
            plt.title('ROC曲線')
            plt.legend()
            plt.savefig(os.path.join(plots_dir, 'roc_curve.png'))
            plt.close()
            
            # 3. Confusion Matrix Heatmap
            cm = confusion_matrix(y_test, (y_pred_proba > 0.5).astype(int))
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('予測ラベル')
            plt.ylabel('実際のラベル')
            plt.title('混同行列')
            plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
            plt.close()
            
            # 4. Baseline vs Optimized Comparison
            if self.baseline_auc and self.optimized_auc:
                plt.figure(figsize=(8, 6))
                models = ['ベースライン', '最適化後']
                aucs = [self.baseline_auc, self.optimized_auc]
                improvement = ((self.optimized_auc - self.baseline_auc) / self.baseline_auc) * 100
                
                sns.barplot(x=models, y=aucs)
                plt.title(f'モデル性能改善 (+{improvement:.2f}%)')
                plt.xlabel('モデル')
                plt.ylabel('AUC スコア')
                plt.ylim(0.5, 1.0)
                
                for i, auc in enumerate(aucs):
                    plt.text(i, auc + 0.01, f'{auc:.4f}', ha='center')
                    
                plt.savefig(os.path.join(plots_dir, 'model_improvement.png'))
                plt.close()
            
            logger.info(f" 可視化を保存しました: {plots_dir}")
            
        except Exception as e:
            logger.error(f" 可視化作成エラー: {e}")
    
    def save_model(self):
        """最適化されたモデルの保存"""
        if self.best_model is None:
            logger.error(" 保存するモデルがありません")
            return None
        
        # Create output directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = os.path.join('models', f'optimized_model_{timestamp}.pkl')
        import joblib
        joblib.dump(self.best_model, model_path)
        
        # Also save as latest model
        latest_path = os.path.join('models', 'optimized_model_latest.pkl')
        joblib.dump(self.best_model, latest_path)
        
        logger.info(f" モデルを保存しました: {model_path}")
        logger.info(f" 最新モデル: {latest_path}")
        
        return model_path
    
    def run_complete_optimization(self):
        """最適化パイプライン全体の実行"""
        logger.info(" モデル最適化パイプラインを開始...")
        
        try:
            # Step 1: Load and prepare data
            X, y = self.load_and_prepare_data()
            
            # Step 2: Evaluate baseline model
            baseline_model, baseline_auc = self.evaluate_baseline_model(X, y)
            
            # Step 3: Select important features
            X_selected, selected_features = self.select_features(X, y)
            
            # Step 4: Handle class imbalance
            X_balanced, y_balanced = self.handle_imbalance(X_selected, y)
            
            # Step 5: Optimize XGBoost
            best_model, best_params, opt_auc = self.optimize_xgboost(X_balanced, y_balanced)
            
            # Step 6: Compare algorithms
            algo_comparison = self.compare_algorithms(X_balanced, y_balanced)
            
            # Step 7: Generate report
            report = self.generate_model_report(X_balanced, y_balanced)
            
            # Step 8: Save model
            model_path = self.save_model()
            
            # Results summary
            improvement = ((opt_auc - baseline_auc) / baseline_auc) * 100
            
            logger.info("=" * 50)
            logger.info(" モデル最適化完了!")
            logger.info(f" AUC改善: {baseline_auc:.4f}  {opt_auc:.4f} (+{improvement:.2f}%)")
            logger.info(f" 特徴量削減: {X.shape[1]}  {X_selected.shape[1]}")
            logger.info(f" 選択された特徴量: {len(selected_features)}")
            logger.info(f" モデル保存先: {model_path}")
            logger.info("=" * 50)
            
            return {
                'baseline_auc': baseline_auc,
                'optimized_auc': opt_auc,
                'improvement': improvement,
                'selected_features': selected_features.tolist(),
                'best_params': best_params,
                'model_path': model_path
            }
            
        except Exception as e:
            logger.error(f" 最適化パイプラインエラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

def main():
    """メイン関数"""
    logger.info(" モデル最適化パイプラインを開始")
    logger.info("=" * 60)
    
    optimizer = ModelOptimizer()
    results = optimizer.run_complete_optimization()
    
    if results:
        logger.info(" 最適化完了")
    else:
        logger.error(" 最適化に失敗しました")
    
    return results

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Streamlit Dashboard for AI Payment Risk Scoring System
Run with: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_preparation import DataPreparator
from src.model_training import ModelTrainer
from src.scoring import RiskScorer
from src.utils import StreamlitDashboard, ResultsExporter
from src.main import PaymentRiskPipeline
import config

# Configure Streamlit page
st.set_page_config(
    page_title="AI支払いリスクスコアリング",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main dashboard application."""
    st.title("🤖 AI支払いリスクスコアリング ダッシュボード")
    st.markdown("### インテリジェント顧客支払いリスク評価システム")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("🎯 ナビゲーション")
    page = st.sidebar.selectbox(
        "ページ選択",
        ["🏠 ホーム", "📊 データ分析", "🤖 モデル訓練", "📈 リスクスコアリング", "📋 結果"]    )
    
    if page == "🏠 ホーム":
        show_home_page()
    elif page == "📊 データ分析":
        show_data_analysis_page()
    elif page == "🤖 モデル訓練":
        show_model_training_page()
    elif page == "📈 リスクスコアリング":
        show_risk_scoring_page()
    elif page == "📋 結果":
        show_results_page()

def show_home_page():
    """Display the home page."""
    st.header("🏠 AI支払いリスクスコアリング へようこそ")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 システム概要
        このAI搭載システムは以下を使用して顧客の支払いリスクを評価します：
        - **高度なMLモデル**: ハイパーパラメータ最適化付きXGBoost
        - **SHAP説明**: 解釈可能なリスク要因分析
        - **リアルタイムスコアリング**: 即座のリスク評価
        - **包括的分析**: 多次元リスク分析
        """)
    
    with col2:
        st.markdown("""
        ### 🔧 主な機能
        - **5段階パイプライン**: 完全なエンドツーエンド処理
        - **インタラクティブダッシュボード**: リアルタイム可視化
        - **リスク分類**: 高/中/低リスク分類
        - **エクスポート機能**: CSV、Excel、レポート生成
        """)
    
    with col3:
        st.markdown("""
        ### 📊 ビジネスインパクト
        - **リスク軽減**: 高リスク顧客の早期識別
        - **意思決定支援**: データ駆動型支払いポリシー
        - **コスト削減**: 支払い不履行の最小化
        - **コンプライアンス**: 透明で説明可能なAI
        """)
    
    st.markdown("---")
    
    # Quick start section
    st.header("🚀 クイックスタート")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 データをアップロード")
        uploaded_file = st.file_uploader(
            "顧客データのExcelファイルを選択してください",
            type=['xlsx', 'xls'],
            help="Excel形式の顧客支払いデータをアップロードしてください"
        )
        
        if uploaded_file is not None:
            st.success("✅ ファイルが正常にアップロードされました！")
            if st.button("🔄 データを処理", type="primary"):
                process_uploaded_data(uploaded_file)
    
    with col2:
        st.subheader("🎲 サンプルデータを試す")
        st.info("システム機能を探索するためのサンプルデータを生成します")
        
        col2a, col2b = st.columns(2)
        with col2a:
            n_customers = st.number_input("顧客数", min_value=100, max_value=10000, value=1000)
        with col2b:
            n_transactions = st.number_input("顧客あたりの取引数", min_value=1, max_value=20, value=5)
        
        if st.button("🎯 サンプルデータを生成・分析", type="secondary"):
            generate_sample_analysis(n_customers, n_transactions)

def show_data_analysis_page():
    """
    日本語データ分析ページを表示
    """
    st.header("📊 データ分析")
    
    # データのロード（既存のデータまたはアップロードされたデータ）
    uploaded_file = st.file_uploader("📁 顧客データをアップロード", type=["xlsx", "xls"], help="顧客データと未払いフラグを含むExcelファイル")
    
    customer_data = None
    
    if uploaded_file:
        with st.spinner("データを処理中..."):
            customer_data = process_uploaded_data(uploaded_file)
            
    # 既存のデータがあるかチェック
    if customer_data is None and os.path.exists("outputs/processed_japanese_data.csv"):
        customer_data = pd.read_csv("outputs/processed_japanese_data.csv")
        st.info("💾 保存済みの日本語データを使用しています")
    
    if customer_data is not None:
        # データの概要
        st.subheader("データサマリー")
        st.write(f"📋 レコード数: {len(customer_data)}")
        
        # 列を表示
        st.write("📊 利用可能な列:")
        st.write(", ".join(customer_data.columns.tolist()))
        
        # タブを作成
        tab1, tab2, tab3 = st.tabs(["📈 基本分析", "💰 支払い分析", "🔍 リスク指標"])
        
        with tab1:
            # 会話時間の分布
            if 'total_conversation_duration' in customer_data.columns:
                st.subheader("会話時間分布")
                fig_duration = px.histogram(
                    customer_data, 
                    x='total_conversation_duration', 
                    title='会話時間分布', 
                    nbins=30
                )
                st.plotly_chart(fig_duration, use_container_width=True)
            
            # 感情スコアの分布
            if 'customer_sentiment_score' in customer_data.columns:
                st.subheader("顧客感情スコア分布")
                fig_sentiment = px.histogram(
                    customer_data, 
                    x='customer_sentiment_score', 
                    title='顧客感情スコア分布', 
                    nbins=20
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
            
            # エージェント発言量と顧客発言量の相関
            if 'agent_total_sentence' in customer_data.columns and 'customer_total_sentence' in customer_data.columns:
                st.subheader("エージェントと顧客の発言量")
                fig_talk = px.scatter(
                    customer_data, 
                    x='agent_total_sentence', 
                    y='customer_total_sentence',
                    title='エージェントと顧客の発言量の関係',
                    labels={'agent_total_sentence': 'エージェント発言数', 'customer_total_sentence': '顧客発言数'}
                )
                st.plotly_chart(fig_talk, use_container_width=True)
        
        with tab2:
            # 未払いフラグの分布
            if '未払FLAG' in customer_data.columns:
                st.subheader("支払いステータス分布")
                payment_counts = customer_data['未払FLAG'].value_counts().reset_index()
                payment_counts.columns = ['支払いステータス', '件数']
                
                fig_payment = px.pie(
                    payment_counts, 
                    values='件数', 
                    names='支払いステータス', 
                    title='支払いステータス分布',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_payment, use_container_width=True)
            
            # 感情スコアと未払いの関係
            if 'customer_sentiment_score' in customer_data.columns and '未払FLAG' in customer_data.columns:
                st.subheader("感情スコアと支払いの関係")
                fig_sentiment_payment = px.box(
                    customer_data, 
                    x='未払FLAG', 
                    y='customer_sentiment_score',
                    title='支払いステータス別の顧客感情スコア',
                    labels={'未払FLAG': '支払いステータス', 'customer_sentiment_score': '顧客感情スコア'}
                )
                st.plotly_chart(fig_sentiment_payment, use_container_width=True)
        
        with tab3:
            # リスク要因の相関マトリックス
            st.subheader("リスク要因の相関関係")
            
            # 数値列のみを選択
            numeric_cols = customer_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            # 相関マトリックスの計算と表示
            if len(numeric_cols) > 1:
                corr_matrix = customer_data[numeric_cols].corr()
                fig_corr = px.imshow(
                    corr_matrix, 
                    title='リスク要因の相関マトリックス',
                    labels=dict(color="相関係数")
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # リスク予測に重要な特徴量
            st.subheader("重要特徴量分析")
            st.write("未払い予測に最も影響する要因:")
            
            if '未払FLAG' in customer_data.columns and len(numeric_cols) > 1:
                from sklearn.ensemble import RandomForestClassifier
                
                # 対象変数と説明変数の準備
                X = customer_data[numeric_cols].drop('未払FLAG', axis=1, errors='ignore')
                if len(X.columns) > 0 and '未払FLAG' in customer_data.columns:
                    y = customer_data['未払FLAG']
                    
                    # ランダムフォレストで特徴量重要度を計算
                    try:
                        model = RandomForestClassifier(n_estimators=50, random_state=42)
                        model.fit(X, y)
                        
                        # 特徴量重要度のデータフレーム作成
                        feature_importance = pd.DataFrame({
                            '特徴量': X.columns,
                            '重要度': model.feature_importances_
                        }).sort_values('重要度', ascending=False)
                        
                        # 重要度のプロット
                        fig_importance = px.bar(
                            feature_importance.head(10), 
                            x='重要度', 
                            y='特徴量',
                            title='トップ10重要特徴量',
                            orientation='h'
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                    except Exception as e:
                        st.error(f"特徴量重要度分析でエラーが発生しました: {e}")
                else:
                    st.warning("数値特徴量が不足しているため、重要度分析を実行できません。")
            else:
                st.warning("未払FLAGまたは数値特徴量が不足しているため、重要度分析を実行できません。")
    else:
        st.info("📁 分析するには顧客データをアップロードしてください")

def show_model_training_page():
    """Display the model training page."""
    st.header("🤖 モデル訓練・評価")
    
    # Check if data exists
    if 'customer_data' not in st.session_state:
        st.warning("⚠️ データが読み込まれていません。まずデータ分析ページからデータを読み込んでください。")
        return
    
    st.subheader("⚙️ 訓練設定")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("テストセットサイズ", 0.1, 0.4, 0.2)
        random_state = st.number_input("ランダムシード", 0, 100, 42)
    
    with col2:
        enable_tuning = st.checkbox("ハイパーパラメータ調整を有効化", value=True)
        cv_folds = st.number_input("交差検証フォールド数", 2, 10, 5)
    
    if st.button("🚀 モデルを訓練", type="primary"):
        train_model_pipeline(test_size, random_state, enable_tuning, cv_folds)
    
    # Display model results if available
    if 'model_metrics' in st.session_state:
        display_model_results()

def show_risk_scoring_page():
    """Display the risk scoring page."""
    st.header("📈 リスクスコアリング・分析")
      # Check if model is trained
    if 'trained_model' not in st.session_state:
        st.warning("⚠️ 訓練されたモデルがありません。まずモデルを訓練してください。")
        return
    
    if st.button("🎯 リスクスコアを生成", type="primary"):
        generate_risk_scores()
    
    # Display risk scores if available
    if 'risk_results' in st.session_state:
        display_risk_results()

def show_results_page():
    """Display the results page."""
    st.header("📋 結果・エクスポート")
    
    # Check if results exist
    if 'risk_results' not in st.session_state:
        st.warning("⚠️ 結果がありません。まずリスクスコアリングプロセスを完了してください。")
        return
    
    risk_results = st.session_state.risk_results
    
    # Results overview
    st.subheader("📊 結果概要")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(risk_results)
        st.metric("総顧客数", total_customers)
    
    with col2:
        high_risk_count = len(risk_results[risk_results['risk_category'] == 'High Risk'])
        st.metric("高リスク顧客数", high_risk_count)
    
    with col3:
        avg_risk_score = risk_results['risk_score'].mean()
        st.metric("平均リスクスコア", f"{avg_risk_score:.2f}")
    
    with col4:
        if 'model_metrics' in st.session_state:
            accuracy = st.session_state.model_metrics.get('accuracy', 0)
            st.metric("モデル精度", f"{accuracy:.3f}")
    
    # Risk distribution
    st.subheader("🎯 リスク分布")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk score histogram
        fig_hist = px.histogram(risk_results, x='risk_score', title='リスクスコア分布', nbins=30)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Risk category pie chart
        risk_counts = risk_results['risk_category'].value_counts()
        fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index, 
                        title='リスクカテゴリー分布')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Export options
    st.subheader("💾 エクスポートオプション")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 リスクスコアをエクスポート (CSV)"):
            csv_data = risk_results.to_csv(index=False)
            st.download_button(
                label="CSVをダウンロード",
                data=csv_data,
                file_name="risk_scores.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 可視化をエクスポート"):
            st.info("可視化エクスポート機能はここに実装されます")
    
    with col3:
        if st.button("📋 レポート生成"):
            st.info("レポート生成機能はここに実装されます")
    
    # Detailed results table
    st.subheader("📋 詳細結果")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        risk_filter = st.selectbox("リスクカテゴリーでフィルター", 
                                  ['すべて'] + list(risk_results['risk_category'].unique()))
    
    with col2:
        score_range = st.slider("リスクスコア範囲",
                               float(risk_results['risk_score'].min()),
                               float(risk_results['risk_score'].max()),
                               (float(risk_results['risk_score'].min()), 
                                float(risk_results['risk_score'].max())))
    
    # Apply filters
    filtered_results = risk_results.copy()
    
    if risk_filter != 'All':
        filtered_results = filtered_results[filtered_results['risk_category'] == risk_filter]
    
    filtered_results = filtered_results[
        (filtered_results['risk_score'] >= score_range[0]) & 
        (filtered_results['risk_score'] <= score_range[1])
    ]
    
    st.dataframe(filtered_results, use_container_width=True)

def process_uploaded_data(uploaded_file):
    """
    アップロードされた日本語のExcelファイルを処理する
    """
    try:
        # ファイル拡張子のチェック
        if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file)
            
            # データの前処理
            from src.data_preparation import DataPreparator
            data_prep = DataPreparator()
            
            # データの検証
            if '未払FLAG' not in df.columns or 'レコード番号' not in df.columns:
                st.error("❌ エラー: ファイルに必要な列（未払FLAG、レコード番号）がありません。")
                return None
                
            # データの前処理と特徴量エンジニアリング
            st.info("🔄 データを処理中...")
            
            # 日本語データ用の特徴エンジニアリングメソッドを使用
            if hasattr(data_prep, 'engineer_features_japanese'):
                processed_df = data_prep.engineer_features_japanese(df)
            else:
                # 通常の特徴エンジニアリングにフォールバック
                processed_df = data_prep.engineer_features_real_data(df)
            
            st.success(f"✅ データ処理が完了しました。{len(processed_df)} 件のレコードが読み込まれました。")
            
            # 処理されたデータの保存
            processed_df.to_csv("outputs/processed_japanese_data.csv", index=False)
            
            return processed_df
            
        else:
            st.error("❌ エラー: サポートされていないファイル形式です。Excel (.xlsx, .xls) ファイルをアップロードしてください。")
            return None
            
    except Exception as e:
        st.error(f"❌ データ処理エラー: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None
def generate_sample_analysis(n_customers, n_transactions):
    """Generate and analyze sample data."""
    try:
        with st.spinner("Generating sample data..."):
            # Initialize data preparator
            data_prep = DataPreparator()
            
            # Generate sample data
            sample_data = data_prep.generate_sample_data(n_customers, n_transactions)
            customer_data = data_prep.aggregate_customer_data(sample_data)
            featured_data = data_prep.engineer_features(customer_data)
            
            # Store in session state
            st.session_state.customer_data = featured_data
            
            st.success(f"✅ Sample data generated! {len(featured_data)} customers created.")
            st.info("👈 Navigate to 'Data Analysis' to explore the generated data.")
            
    except Exception as e:
        st.error(f"❌ Error generating sample data: {str(e)}")

def train_model_pipeline(test_size, random_state, enable_tuning, cv_folds):
    """Train the ML model."""
    try:
        with st.spinner("Training model..."):
            customer_data = st.session_state.customer_data
            
            # Initialize components
            data_prep = DataPreparator()
            model_trainer = ModelTrainer()
            
            # Prepare ML data
            X, y, feature_names = data_prep.prepare_ml_data(customer_data)
            
            # Train model
            results = model_trainer.train_model(
                X, y, feature_names,
                test_size=test_size,
                random_state=random_state,
                enable_hyperparameter_tuning=enable_tuning,
                cv_folds=cv_folds
            )
            
            # Store results
            st.session_state.model_metrics = results
            st.session_state.trained_model = model_trainer.model
            st.session_state.feature_names = feature_names
            
            st.success("✅ Model trained successfully!")
            
    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")

def display_model_results():
    """Display model training results."""
    metrics = st.session_state.model_metrics
    
    st.subheader("📊 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    with col2:
        st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
    with col3:
        st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
    with col4:
        st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
    
    # Feature importance
    if 'feature_importance' in metrics:
        st.subheader("🎯 Feature Importance")
        importance_data = metrics['feature_importance']
        if isinstance(importance_data, dict):
            importance_df = pd.DataFrame(list(importance_data.items()), 
                                       columns=['Feature', 'Importance'])
            importance_df = importance_df.sort_values('Importance', ascending=True)
            
            fig_importance = px.bar(importance_df.tail(15), x='Importance', y='Feature', 
                                  orientation='h', title='Top 15 Feature Importance')
            st.plotly_chart(fig_importance, use_container_width=True)

def generate_risk_scores():
    """Generate risk scores for customers."""
    try:
        with st.spinner("Generating risk scores..."):
            customer_data = st.session_state.customer_data
            model = st.session_state.trained_model
            feature_names = st.session_state.feature_names
            
            # Initialize risk scorer
            risk_scorer = RiskScorer()
            risk_scorer.model = model
            
            # Prepare data
            data_prep = DataPreparator()
            X, y, _ = data_prep.prepare_ml_data(customer_data)
            
            # Generate scores
            risk_results = risk_scorer.score_customers(X, feature_names, customer_data)
            
            # Store results
            st.session_state.risk_results = risk_results
            
            st.success(f"✅ Risk scores generated for {len(risk_results)} customers!")
            
    except Exception as e:
        st.error(f"❌ Error generating risk scores: {str(e)}")

def display_risk_results():
    """Display risk scoring results."""
    risk_results = st.session_state.risk_results
    
    st.subheader("📈 Risk Scoring Results")
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_score = risk_results['risk_score'].mean()
        st.metric("Average Risk Score", f"{avg_score:.2f}")
    
    with col2:
        high_risk_pct = (risk_results['risk_category'] == 'High Risk').mean() * 100
        st.metric("High Risk %", f"{high_risk_pct:.1f}%")
    
    with col3:
        score_std = risk_results['risk_score'].std()
        st.metric("Score Std Dev", f"{score_std:.2f}")
    
    # Top risk customers
    st.subheader("⚠️ Top Risk Customers")
    top_risk = risk_results.nlargest(10, 'risk_score')[['customer_id', 'risk_score', 'risk_category', 'shap_explanation']]
    st.dataframe(top_risk, use_container_width=True)

if __name__ == "__main__":
    main()

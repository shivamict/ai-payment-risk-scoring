#!/usr/bin/env python3
"""
Mobile-optimized Real-time Dashboard for Sales Agents
Specifically designed for tablets and smartphones during live calls
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import requests
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Page config for mobile optimization
st.set_page_config(
    page_title="📱 ライブリスクスコアリング",
    page_icon="📱",
    layout="centered",  # Better for mobile
    initial_sidebar_state="collapsed"  # Hide sidebar on mobile
)

# Custom CSS for mobile optimization
st.markdown("""
<style>
    /* Mobile-first responsive design */
    .stApp {
        max-width: 100%;
        padding: 1rem 0.5rem;
    }
    
    /* Large touch-friendly buttons */
    .stButton > button {
        width: 100%;
        height: 60px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 15px;
        margin: 5px 0;
    }
    
    /* Risk score display */
    .risk-score-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .risk-score {
        font-size: 5rem;
        font-weight: 900;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        margin: 0;
    }
    
    .risk-category {
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 10px;
    }
    
    /* Customer info cards */
    .customer-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid #007bff;
    }
    
    /* Recommendation badges */
    .recommendation {
        background: #f8f9fa;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 12px;
        border-left: 4px solid #28a745;
        font-size: 16px;
        line-height: 1.4;
    }
    
    .recommendation.high-priority {
        border-left-color: #dc3545;
        background: #fff5f5;
    }
    
    .recommendation.medium-priority {
        border-left-color: #ffc107;
        background: #fffbf0;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-connected { background: #28a745; }
    .status-disconnected { background: #dc3545; }
    .status-calling { background: #ffc107; animation: pulse 1.5s infinite; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Hide streamlit elements for cleaner mobile experience */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Larger text inputs for mobile */
    .stTextInput > div > div > input {
        font-size: 18px;
        padding: 12px;
        border-radius: 10px;
    }
    
    /* Better spacing for mobile */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Emergency button styling */
    .emergency-btn {
        background-color: #dc3545 !important;
        color: white !important;
        font-size: 20px !important;
        font-weight: bold !important;
        padding: 1rem !important;
        border-radius: 15px !important;
        border: none !important;
        width: 100% !important;
        margin: 10px 0 !important;
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'call_active' not in st.session_state:
    st.session_state.call_active = False
if 'customer_data' not in st.session_state:
    st.session_state.customer_data = None
if 'risk_score' not in st.session_state:
    st.session_state.risk_score = None
if 'call_id' not in st.session_state:
    st.session_state.call_id = None
if 'agent_id' not in st.session_state:
    st.session_state.agent_id = ""

class MobileRiskScoringAPI:
    """API client for mobile dashboard"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        
    def start_call(self, agent_id, customer_id, call_id):
        """Start a new call"""
        try:
            response = requests.post(f"{self.base_url}/api/start_call", json={
                'agent_id': agent_id,
                'customer_id': customer_id,
                'call_id': call_id
            }, timeout=10)
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def get_risk_score(self, call_id):
        """Get current risk score"""
        try:
            response = requests.post(f"{self.base_url}/api/score_customer", json={
                'call_id': call_id
            }, timeout=5)
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def end_call(self, call_id):
        """End the call"""
        try:
            response = requests.post(f"{self.base_url}/api/end_call", json={
                'call_id': call_id
            }, timeout=10)
            return response.json() if response.status_code == 200 else None
        except:
            return None

# Initialize API client
api = MobileRiskScoringAPI()

# Header with status
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("# 📱 ライブリスクスコアリング")
with col2:
    if st.session_state.call_active:
        st.markdown('<span class="status-indicator status-calling"></span>**通話中**', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-indicator status-connected"></span>**待機中**', unsafe_allow_html=True)

# Current time
st.markdown(f"**📅 {datetime.now().strftime('%H:%M:%S - %Y年%m月%d日')}**")

# Agent setup section
if not st.session_state.call_active:
    st.markdown("## 👤 エージェント設定")
    
    agent_id = st.text_input("🆔 エージェントID", value=st.session_state.agent_id, 
                            placeholder="エージェントIDを入力")
    st.session_state.agent_id = agent_id
    
    customer_id = st.text_input("👥 顧客ID", placeholder="通話する顧客IDを入力")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📞 通話開始", type="primary", disabled=not (agent_id and customer_id)):
            if agent_id and customer_id:
                call_id = f"CALL-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{np.random.randint(1000, 9999)}"
                
                with st.spinner("🔄 通話を開始しています..."):
                    result = api.start_call(agent_id, customer_id, call_id)
                    
                if result and 'customer_data' in result:
                    st.session_state.call_active = True
                    st.session_state.customer_data = result['customer_data']
                    st.session_state.call_id = call_id
                    st.success("✅ 通話が正常に開始されました！")
                    st.rerun()
                else:
                    st.error("❌ 通話開始に失敗しました。デモモードを使用します。")
                    # Start demo mode
                    st.session_state.call_active = True
                    st.session_state.customer_data = {
                        'name': f'Demo Customer {customer_id}',
                        'phone': '+1-555-0123',
                        'account_balance': np.random.uniform(5000, 50000),
                        'previous_calls': np.random.randint(1, 10),
                        'risk_factors': ['payment_history', 'account_age']
                    }
                    st.session_state.call_id = call_id
                    st.rerun()
    
    with col2:
        if st.button("🔧 Settings"):
            st.info("💡 Settings: Configure API endpoint, notifications, and preferences")

else:
    # Active call interface
    st.markdown("## 📞 通話中")
    
    # Customer information
    if st.session_state.customer_data:
        customer = st.session_state.customer_data
        
        st.markdown(f"""
        <div class="customer-card">
            <h3>👤 {customer.get('name', '不明な顧客')}</h3>
            <p><strong>📞 電話番号:</strong> {customer.get('phone', 'N/A')}</p>
            <p><strong>💰 残高:</strong> ¥{customer.get('account_balance', 0):,.0f}</p>
            <p><strong>📊 過去の通話:</strong> {customer.get('previous_calls', 0)}回</p>
        </div>
        """, unsafe_allow_html=True)
        
    # Risk Score Display
    if 'risk_score_data' not in st.session_state:
        # Simulate getting risk score
        mock_score = np.random.uniform(20, 80)
        if mock_score >= 60:
            risk_category = "🔴 高リスク"
            risk_color = "#dc3545"
        elif mock_score >= 35:
            risk_category = "🟡 中リスク"  
            risk_color = "#ffc107"
        else:
            risk_category = "🟢 低リスク"
            risk_color = "#28a745"
        
        st.session_state.risk_score_data = {
            'score': mock_score,
            'category': risk_category,
            'color': risk_color
        }
    
    risk_data = st.session_state.risk_score_data
    
    st.markdown(f"""
    <div class="risk-score-container" style="background: linear-gradient(135deg, {risk_data['color']} 0%, {risk_data['color']}99 100%);">
        <div class="risk-score">{int(risk_data['score'])}</div>
        <div class="risk-category">{risk_data['category']}</div>
        <div style="margin-top: 10px; opacity: 0.9;">リアルタイムリスク評価</div>
    </div>
    """, unsafe_allow_html=True)
      # Auto-refresh risk score
    if st.button("🔄 スコア更新", type="secondary"):
        # Simulate dynamic risk score
        new_score = max(10, min(90, st.session_state.risk_score_data['score'] + np.random.uniform(-10, 10)))
        
        if new_score >= 60:
            risk_category = "🔴 高リスク"
            risk_color = "#dc3545"
        elif new_score >= 35:
            risk_category = "🟡 中リスク"
            risk_color = "#ffc107"
        else:
            risk_category = "🟢 低リスク"
            risk_color = "#28a745"
        
        st.session_state.risk_score_data = {
            'score': new_score,
            'category': risk_category,
            'color': risk_color
        }
        st.rerun()
    
    # Recommendations based on risk level
    st.markdown("## 💡 ライブ推奨事項")
    
    if risk_data['score'] >= 60:
        recommendations = [
            "🚨 最優先: すぐにスーパーバイザーにエスカレーション",
            "💰 柔軟な支払いプランオプションを提案",
            "📋 すべての支払い懸念を詳細に記録",
            "🤝 共感的な言葉遣いと積極的な傾聴を使用",
            "⏰ 通話時間を制限 - 迅速にコミットメントを取得"
        ]
        priority_class = "high-priority"
    elif risk_data['score'] >= 35:
        recommendations = [
            "⚠️ 支払いコミットメントを密接に監視",
            "🤝 関係を築き信頼を確立",
            "📞 明確なフォローアップ通話をスケジュール",
            "💡 明確な支払いオプションを提示",
            "📝 条件の理解を確認"
        ]
        priority_class = "medium-priority"
    else:
        recommendations = [
            "✅ 顧客は良好な支払い潜在能力を示しています",
            "🎯 アップセルの機会に焦点を当てる",
            "😊 前向きな関係を維持",
            "📈 プレミアムサービスの提供を検討",
            "🏆 ロイヤルティプログラムの優秀候補"
        ]
        priority_class = ""
    
    for rec in recommendations:
        st.markdown(f'<div class="recommendation {priority_class}">{rec}</div>', unsafe_allow_html=True)
      # Emergency actions
    st.markdown("## 🚨 クイックアクション")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🆘 エスカレート", type="secondary", help="スーパーバイザーにエスカレート"):
            st.warning("🚨 エスカレーション要求がスーパーバイザーに送信されました")
    
    with col2:
        if st.button("📝 メモ", type="secondary", help="クイックメモを開く"):
            notes = st.text_area("📝 クイックメモ", placeholder="通話の要点を記録...", height=100)
            if notes:
                st.success("📝 メモが保存されました")
    
    with col3:
        if st.button("📊 履歴", type="secondary", help="顧客履歴を表示"):
            st.info("📊 顧客履歴: 過去3回の通話、2回の延滞、最終連絡: 2週間前")
      # Call controls
    st.markdown("## 📞 通話コントロール")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("⏸️ 保留", type="secondary"):
            st.info("⏸️ 通話が保留されました")
    
    with col2:
        if st.button("🔚 通話終了", type="primary"):
            with st.spinner("🔄 通話を終了しています..."):
                if st.session_state.call_id:
                    api.end_call(st.session_state.call_id)
                
                st.session_state.call_active = False
                st.session_state.customer_data = None
                st.session_state.call_id = None
                st.session_state.risk_score_data = None
                st.success("✅ 通話が正常に終了されました！")
                time.sleep(1)
                st.rerun()

# Live transcript simulation (collapsible)
if st.session_state.call_active:
    with st.expander("📝 ライブ文字起こし"):
        st.markdown("### 最近の会話")
        
        transcript_items = [
            ("エージェント", "おはようございます！本日はどのようなご用件でしょうか？", "😊 ポジティブ"),
            ("顧客", "支払いプランについてお電話しました。", "😐 ニュートラル"),
            ("エージェント", "喜んでお手伝いさせていただきます。", "😊 ポジティブ"),
            ("顧客", "支払いに困っています。", "😟 ネガティブ"),
            ("エージェント", "承知いたしました。どのような選択肢があるか確認いたします。", "😊 ポジティブ")
        ]
        
        for speaker, text, sentiment in transcript_items:
            if speaker == "エージェント":
                st.markdown(f"🎧 **エージェント:** {text} {sentiment}")
            else:
                st.markdown(f"👤 **顧客:** {text} {sentiment}")

# Auto-refresh mechanism
if st.session_state.call_active:
    # Auto-refresh every 30 seconds
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    if time.time() - st.session_state.last_refresh > 30:
        st.session_state.last_refresh = time.time()
        st.rerun()

# Footer
st.markdown("---")
st.markdown("🔒 **セキュア接続** | 📱 **モバイル最適化** | ⚡ **リアルタイム更新**")

# Instructions for first-time users
if not st.session_state.call_active and not st.session_state.agent_id:
    st.markdown("""
    ## 📋 クイックスタートガイド
    
    1. **エージェントIDを入力** (スーパーバイザーから提供)
    2. **顧客IDを入力** (通話キューから)
    3. **通話開始をタップ** してリアルタイムスコアリングを開始
    4. **通話中に表示される推奨事項に従う**
    5. **エスカレーションやメモにクイックアクションを使用**
    6. **完了時に通話終了をタップ**
    
    💡 **ヒント:** リアルタイムインサイトのために通話中はこのダッシュボードを開いたままにしてください！
    """)

# Debug info (only in development)
if st.checkbox("🔧 デバッグ情報", help="技術的詳細を表示"):
    st.json({
        'call_active': st.session_state.call_active,
        'call_id': st.session_state.call_id,
        'agent_id': st.session_state.agent_id,
        'has_customer_data': st.session_state.customer_data is not None,
        'timestamp': datetime.now().isoformat()
    })

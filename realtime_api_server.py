#!/usr/bin/env python3
"""
Real-time Customer Risk Scoring API Server
Integrates with Amazon Connect and Kintone for live call scoring
"""

import os
import sys
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import joblib
import boto3
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import requests

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.scoring import RiskScorer
from src.data_preparation import DataPreparator
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AmazonConnectIntegration:
    """Integration with Amazon Connect for live call processing"""
    
    def __init__(self, region_name='us-east-1'):
        self.region_name = region_name
        self.connect_client = None
        self.transcribe_client = None
        self.active_calls = {}
        
    def initialize_aws_clients(self, aws_access_key_id=None, aws_secret_access_key=None):
        """Initialize AWS clients for Connect and Transcribe"""
        try:
            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=self.region_name
            )
            
            self.connect_client = session.client('connect')
            self.transcribe_client = session.client('transcribe')
            logger.info("AWS clients initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
            return False
    
    async def start_call_monitoring(self, call_id: str, contact_id: str):
        """Start monitoring a live call for real-time transcription"""
        try:
            # Store call information
            self.active_calls[call_id] = {
                'contact_id': contact_id,
                'start_time': datetime.now(),
                'transcripts': [],
                'sentiment_scores': [],
                'is_active': True
            }
            
            # Start real-time transcription (mock implementation)
            logger.info(f"Started monitoring call {call_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start call monitoring: {e}")
            return False
    
    async def get_live_transcript(self, call_id: str) -> Dict[str, Any]:
        """Get live transcript data from ongoing call"""
        if call_id not in self.active_calls:
            return {'error': 'Call not found'}
        
        # Mock live transcript data (replace with actual Connect integration)
        mock_transcript = {
            'call_id': call_id,
            'timestamp': datetime.now().isoformat(),
            'agent_text': "How can I help you today?",
            'customer_text': "I need help with my payment plan",
            'agent_sentiment': 0.8,
            'customer_sentiment': -0.2,
            'confidence': 0.95
        }
        
        self.active_calls[call_id]['transcripts'].append(mock_transcript)
        return mock_transcript
    
    def stop_call_monitoring(self, call_id: str):
        """Stop monitoring a call"""
        if call_id in self.active_calls:
            self.active_calls[call_id]['is_active'] = False
            logger.info(f"Stopped monitoring call {call_id}")

class KintoneIntegration:
    """Integration with Kintone for customer data retrieval"""
    
    def __init__(self, subdomain: str, app_id: str, api_token: str):
        self.subdomain = subdomain
        self.app_id = app_id
        self.api_token = api_token
        self.base_url = f"https://{subdomain}.cybozu.com/k/v1/"
        
    def get_customer_data(self, customer_id: str) -> Dict[str, Any]:
        """Retrieve customer data from Kintone"""
        try:
            headers = {
                'X-Cybozu-API-Token': self.api_token,
                'Content-Type': 'application/json'
            }
            
            # Query customer data
            query = f'customer_id = "{customer_id}"'
            url = f"{self.base_url}records.json"
            params = {
                'app': self.app_id,
                'query': query
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('records'):
                    return self._format_customer_data(data['records'][0])
                else:
                    return {'error': 'Customer not found'}
            else:
                logger.error(f"Kintone API error: {response.status_code}")
                return {'error': 'API error'}
                
        except Exception as e:
            logger.error(f"Failed to retrieve customer data: {e}")
            # Return mock data for development
            return self._get_mock_customer_data(customer_id)
    
    def _format_customer_data(self, record: Dict) -> Dict[str, Any]:
        """Format Kintone record to standard customer data format"""
        return {
            'customer_id': record.get('customer_id', {}).get('value', ''),
            'name': record.get('name', {}).get('value', ''),
            'phone': record.get('phone', {}).get('value', ''),
            'email': record.get('email', {}).get('value', ''),
            'account_balance': float(record.get('account_balance', {}).get('value', 0)),
            'payment_history': record.get('payment_history', {}).get('value', []),
            'risk_factors': record.get('risk_factors', {}).get('value', [])
        }
    
    def _get_mock_customer_data(self, customer_id: str) -> Dict[str, Any]:
        """Return mock customer data for development/testing"""
        return {
            'customer_id': customer_id,
            'name': f'Customer {customer_id}',
            'phone': '+1-555-0123',
            'email': f'customer{customer_id}@example.com',
            'account_balance': np.random.uniform(1000, 50000),
            'payment_history': ['on_time', 'late', 'on_time', 'on_time'],
            'risk_factors': ['high_balance', 'irregular_payments'],
            'previous_calls': np.random.randint(1, 10),
            'satisfaction_score': np.random.uniform(3.0, 5.0)
        }

class RealTimeRiskScorer:
    """Real-time risk scoring engine"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_trained_model()
        
    def load_trained_model(self):
        """Load the trained model and preprocessing components"""
        try:
            # Load the latest trained model
            models_dir = config.MODELS_DIR
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
            
            if model_files:
                latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
                model_path = os.path.join(models_dir, latest_model)
                
                model_data = joblib.load(model_path)
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.feature_names = model_data.get('feature_names', [])
                
                logger.info(f"Loaded model: {latest_model}")
            else:
                logger.warning("No trained model found. Using mock scoring.")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def score_customer_realtime(self, call_data: Dict, customer_data: Dict) -> Dict[str, Any]:
        """Generate real-time risk score based on live call and customer data"""
        try:
            # Extract features from live call and customer data
            features = self._extract_realtime_features(call_data, customer_data)
            
            if self.model and self.scaler:
                # Use trained model for scoring
                features_scaled = self.scaler.transform([features])
                risk_score = self.model.predict_proba(features_scaled)[0][1] * 100
            else:
                # Mock scoring for development
                risk_score = self._calculate_mock_risk_score(call_data, customer_data)
            
            # Determine risk category
            risk_category = self._categorize_risk(risk_score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(risk_score, call_data, customer_data)
            
            return {
                'customer_id': customer_data.get('customer_id'),
                'risk_score': round(risk_score, 2),
                'risk_category': risk_category,
                'confidence': 0.85,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat(),
                'factors': self._get_risk_factors(call_data, customer_data)
            }
            
        except Exception as e:
            logger.error(f"Error in real-time scoring: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _extract_realtime_features(self, call_data: Dict, customer_data: Dict) -> List[float]:
        """Extract features for real-time scoring"""
        features = []
        
        # Call-based features
        features.extend([
            call_data.get('agent_sentiment', 0.0),
            call_data.get('customer_sentiment', 0.0),
            call_data.get('call_duration', 0.0),
            call_data.get('agent_talk_ratio', 0.5),
            call_data.get('silence_ratio', 0.1)
        ])
        
        # Customer-based features
        features.extend([
            customer_data.get('account_balance', 0.0),
            customer_data.get('previous_calls', 0),
            customer_data.get('satisfaction_score', 3.0),
            len(customer_data.get('payment_history', [])),
            len(customer_data.get('risk_factors', []))
        ])
        
        # Ensure we have the right number of features
        while len(features) < 25:  # Pad to expected feature count
            features.append(0.0)
        
        return features[:25]  # Truncate to expected feature count
    
    def _calculate_mock_risk_score(self, call_data: Dict, customer_data: Dict) -> float:
        """Calculate mock risk score for development"""
        base_score = 50.0
        
        # Adjust based on sentiment
        customer_sentiment = call_data.get('customer_sentiment', 0.0)
        if customer_sentiment < -0.5:
            base_score += 20
        elif customer_sentiment < 0:
            base_score += 10
          # Adjust based on account balance
        balance = customer_data.get('account_balance', 0)
        if balance < 5000:
            base_score += 15
        elif balance > 30000:
            base_score -= 10
        
        # Adjust based on risk factors
        risk_factors = len(customer_data.get('risk_factors', []))
        base_score += risk_factors * 5
        
        return max(0, min(100, base_score))
    
    def _categorize_risk(self, risk_score: float) -> str:
        """リスクスコアを高/中/低に分類"""
        if risk_score >= 70:
            return "高リスク"
        elif risk_score >= 40:
            return "中リスク"
        else:
            return "低リスク"
    
    def _generate_recommendations(self, risk_score: float, call_data: Dict, customer_data: Dict) -> List[str]:
        """エージェントに対する実用的な推奨事項を生成"""
        recommendations = []
        
        if risk_score >= 70:
            recommendations.extend([
                "🚨 高リスク顧客 - 上司にエスカレーションしてください",
                "💰 支払いプランのオプションを提示してください",
                "📋 支払いに関する懸念を詳細に記録してください"
            ])
        elif risk_score >= 40:
            recommendations.extend([
                "⚠️ 支払い約束を注意深く監視してください",
                "🤝 信頼関係を築いてください",
                "📞 フォローアップコールをスケジュールしてください"
            ])
        else:
            recommendations.extend([
                "✅ 顧客は良好な支払い可能性を示しています",
                "🎯 アップセリング機会に焦点を当ててください",
                "😊 良好な関係を維持してください"
            ])
        
        # 感情に基づく推奨事項を追加
        customer_sentiment = call_data.get('customer_sentiment', 0.0)
        if customer_sentiment < -0.3:
            recommendations.append("😟 顧客が不満を感じているようです - 懸念事項に対処してください")
        
        return recommendations
    
    def _get_risk_factors(self, call_data: Dict, customer_data: Dict) -> List[str]:
        """主要なリスク要因を特定"""
        factors = []
        
        if call_data.get('customer_sentiment', 0) < -0.3:
            factors.append("顧客感情がネガティブ")
        
        if customer_data.get('account_balance', 0) < 5000:
            factors.append("口座残高が低い")
        
        if len(customer_data.get('risk_factors', [])) > 2:
            factors.append("複数の過去のリスク要因")
        
        return factors

# Initialize Flask app and components
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize components
amazon_connect = AmazonConnectIntegration()
kintone = KintoneIntegration(
    subdomain=os.getenv('KINTONE_SUBDOMAIN', 'demo'),
    app_id=os.getenv('KINTONE_APP_ID', '1'),
    api_token=os.getenv('KINTONE_API_TOKEN', 'demo-token')
)
risk_scorer = RealTimeRiskScorer()

# Active connections tracking
active_agents = {}
active_calls = {}

@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('realtime_dashboard.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_calls': len(active_calls),
        'active_agents': len(active_agents)
    })

@app.route('/api/start_call', methods=['POST'])
def start_call():
    """Start monitoring a new call"""
    try:
        data = request.json
        call_id = data.get('call_id')
        customer_id = data.get('customer_id')
        agent_id = data.get('agent_id')
        
        if not all([call_id, customer_id, agent_id]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Get customer data from Kintone
        customer_data = kintone.get_customer_data(customer_id)
        
        # Start Amazon Connect monitoring (using synchronous call)
        success = asyncio.run(amazon_connect.start_call_monitoring(call_id, call_id))
        
        if success:
            # Store call information
            active_calls[call_id] = {
                'customer_id': customer_id,
                'agent_id': agent_id,
                'start_time': datetime.now(),
                'customer_data': customer_data,
                'risk_scores': []
            }
            
            # Notify agent via WebSocket
            socketio.emit('call_started', {
                'call_id': call_id,
                'customer_data': customer_data
            }, room=f'agent_{agent_id}')
            
            return jsonify({
                'status': 'success',
                'call_id': call_id,
                'customer_data': customer_data
            })
        else:
            return jsonify({'error': 'Failed to start call monitoring'}), 500
            
    except Exception as e:
        logger.error(f"Error starting call: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/score_customer', methods=['POST'])
def score_customer():
    """Generate real-time risk score"""
    try:
        data = request.json
        call_id = data.get('call_id')
        
        if call_id not in active_calls:
            return jsonify({'error': 'Call not found'}), 404
        
        # Get live call data
        call_data = asyncio.run(amazon_connect.get_live_transcript(call_id))
        customer_data = active_calls[call_id]['customer_data']
        
        # Generate risk score
        risk_result = risk_scorer.score_customer_realtime(call_data, customer_data)
        
        # Store score history
        active_calls[call_id]['risk_scores'].append(risk_result)
        
        # Emit to agent via WebSocket
        agent_id = active_calls[call_id]['agent_id']
        socketio.emit('risk_score_update', risk_result, room=f'agent_{agent_id}')
        
        return jsonify(risk_result)
        
    except Exception as e:
        logger.error(f"Error scoring customer: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/end_call', methods=['POST'])
def end_call():
    """End call monitoring"""
    try:
        data = request.json
        call_id = data.get('call_id')
        
        if call_id in active_calls:
            # Stop monitoring
            amazon_connect.stop_call_monitoring(call_id)
            
            # Generate call summary
            call_summary = {
                'call_id': call_id,
                'duration': (datetime.now() - active_calls[call_id]['start_time']).seconds,
                'risk_scores': active_calls[call_id]['risk_scores'],
                'final_risk_score': active_calls[call_id]['risk_scores'][-1] if active_calls[call_id]['risk_scores'] else None
            }
            
            # Notify agent
            agent_id = active_calls[call_id]['agent_id']
            socketio.emit('call_ended', call_summary, room=f'agent_{agent_id}')
            
            # Clean up
            del active_calls[call_id]
            
            return jsonify({'status': 'success', 'summary': call_summary})
        else:
            return jsonify({'error': 'Call not found'}), 404
            
    except Exception as e:
        logger.error(f"Error ending call: {e}")
        return jsonify({'error': str(e)}), 500

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle agent connection"""
    logger.info(f"Agent connected: {request.sid}")

@socketio.on('join_agent')
def handle_join_agent(data):
    """Agent joins their room for real-time updates"""
    agent_id = data.get('agent_id')
    if agent_id:
        room = f'agent_{agent_id}'
        join_room(room)
        active_agents[agent_id] = request.sid
        logger.info(f"Agent {agent_id} joined room {room}")
        
        emit('joined', {'room': room, 'agent_id': agent_id})

@socketio.on('leave_agent')
def handle_leave_agent(data):
    """Agent leaves their room"""
    agent_id = data.get('agent_id')
    if agent_id:
        room = f'agent_{agent_id}'
        leave_room(room)
        if agent_id in active_agents:
            del active_agents[agent_id]
        logger.info(f"Agent {agent_id} left room {room}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle agent disconnection"""
    logger.info(f"Agent disconnected: {request.sid}")
    # Clean up agent from active list
    for agent_id, sid in list(active_agents.items()):
        if sid == request.sid:
            del active_agents[agent_id]
            break

# Background task for continuous scoring
def background_scoring():
    """Background task to continuously score active calls"""
    while True:
        try:
            for call_id, call_info in list(active_calls.items()):
                # Get fresh call data and generate new score
                if datetime.now() - call_info['start_time'] > timedelta(seconds=30):
                    # Generate periodic risk update
                    call_data = amazon_connect.get_live_transcript(call_id)
                    if isinstance(call_data, dict) and 'error' not in call_data:
                        risk_result = risk_scorer.score_customer_realtime(
                            call_data, call_info['customer_data']
                        )
                        
                        # Emit update to agent
                        agent_id = call_info['agent_id']
                        socketio.emit('risk_score_update', risk_result, room=f'agent_{agent_id}')
                        
                        # Store in history
                        call_info['risk_scores'].append(risk_result)
            
            time.sleep(10)  # Update every 10 seconds
            
        except Exception as e:
            logger.error(f"Background scoring error: {e}")
            time.sleep(5)

if __name__ == '__main__':
    # Start background scoring task
    scoring_thread = threading.Thread(target=background_scoring, daemon=True)
    scoring_thread.start()
    
    # Run the server
    logger.info("🚀 Starting Real-time Risk Scoring Server...")
    logger.info("📊 Dashboard available at: http://localhost:5000")
    logger.info("🔌 WebSocket enabled for real-time updates")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

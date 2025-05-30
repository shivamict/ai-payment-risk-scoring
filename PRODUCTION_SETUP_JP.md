# 🚀 AI顧客リスクスコアリングシステム - 本番環境セットアップガイド

リアルタイム顧客リスク評価システムをAmazon ConnectとKintone統合で本番環境にデプロイするための包括的なガイドです。

## 📋 本番環境セットアップチェックリスト

### 1. **AWSインフラストラクチャ設定**

#### A. **Amazon Connect設定**

```powershell
# AWS CLIのインストール
pip install awscli boto3

# AWS認証設定
aws configure
```

**AWSリソースの作成:**

```python
"""
本番環境用AWS インフラストラクチャ設定
"""
import boto3
import json
import logging

class AWSProductionSetup:
    def __init__(self, region='us-east-1'):
        self.region = region
        self.connect_client = boto3.client('connect', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.iam_client = boto3.client('iam', region_name=region)
        
    def setup_connect_instance(self):
        """Amazon Connectインスタンスをコール処理用にセットアップ"""
        try:
            # Connectインスタンスの作成
            response = self.connect_client.create_instance(
                IdentityManagementType='CONNECT_MANAGED',
                InstanceAlias='jp-risk-scoring-connect',
                DirectoryId='your-directory-id',  # ディレクトリIDに置換
                InboundCallsEnabled=True,
                OutboundCallsEnabled=True
            )
            
            instance_id = response['Id']
            print(f"✅ Connectインスタンス作成: {instance_id}")
            
            # リアルタイムストリーミングの設定
            self.setup_real_time_streaming(instance_id)
            
            return instance_id
            
        except Exception as e:
            print(f"❌ Connectインスタンス作成エラー: {e}")
            return None
    
    def setup_real_time_streaming(self, instance_id):
        """リアルタイム音声ストリーミングの設定"""
        try:
            response = self.connect_client.put_streaming_configuration(
                InstanceId=instance_id,
                StreamingConfiguration={
                    'DataRetentionPeriod': 7,
                    'EncryptionConfiguration': {
                        'EncryptionType': 'KMS',
                        'KeyId': 'alias/aws/connect'
                    }
                }
            )
            print("✅ リアルタイムストリーミング設定完了")
            
        except Exception as e:
            print(f"❌ ストリーミング設定エラー: {e}")
    
    def create_lambda_functions(self):
        """Connect統合用Lambda関数の作成"""
        
        # リスクスコアリングLambda関数
        lambda_code = '''
import json
import boto3
import requests
import logging

def lambda_handler(event, context):
    """Amazon Connectイベントを処理してリスクスコアリングをトリガー"""
    
    try:
        # Connectイベントからコールデータを抽出
        detail = event.get('detail', {})
        contact_id = detail.get('contactId')
        
        if detail.get('eventType') == 'AGENT_CONNECTED':
            # コール開始 - リスクスコアリングをトリガー
            scoring_api_url = f"https://your-domain.com/api/score_call"
            
            payload = {
                'call_id': contact_id,
                'call_data': {
                    'customer_id': detail.get('attributes', {}).get('customer_id'),
                    'agent_id': detail.get('agentId'),
                    'start_time': detail.get('eventTimestamp')
                }
            }
            
            response = requests.post(scoring_api_url, json=payload)
            
            return {
                'statusCode': 200,
                'body': json.dumps('リスクスコアリング開始')
            }
            
    except Exception as e:
        logging.error(f"Connectイベント処理エラー: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'エラー: {str(e)}')
        }
'''
        
        # Lambda関数の作成
        try:
            response = self.lambda_client.create_function(
                FunctionName='jp-risk-scoring-connect-handler',
                Runtime='python3.9',
                Role='arn:aws:iam::YOUR_ACCOUNT:role/lambda-execution-role',
                Handler='index.lambda_handler',
                Code={'ZipFile': lambda_code.encode()},
                Description='日本語顧客向けリスクスコアリングConnect統合',
                Timeout=30
            )
            
            print(f"✅ Lambda関数作成: {response['FunctionArn']}")
            return response['FunctionArn']
            
        except Exception as e:
            print(f"❌ Lambda関数作成エラー: {e}")
            return None

if __name__ == "__main__":
    setup = AWSProductionSetup()
    
    # Connectインスタンスの作成
    instance_id = setup.setup_connect_instance()
    
    # Lambda関数の作成
    lambda_arn = setup.create_lambda_functions()
    
    print("\\n🎯 AWS セットアップ完了!")
    print(f"Connect インスタンスID: {instance_id}")
    print(f"Lambda 関数ARN: {lambda_arn}")
```

#### B. **環境設定**

```python
"""
本番環境設定
"""
import os
from dataclasses import dataclass

@dataclass
class JapaneseProductionConfig:
    # AWS設定
    AWS_REGION: str = "ap-northeast-1"  # 東京リージョン
    CONNECT_INSTANCE_ID: str = os.getenv("CONNECT_INSTANCE_ID", "")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    
    # Kintone設定
    KINTONE_SUBDOMAIN: str = os.getenv("KINTONE_SUBDOMAIN", "")
    KINTONE_APP_ID: str = os.getenv("KINTONE_APP_ID", "")
    KINTONE_API_TOKEN: str = os.getenv("KINTONE_API_TOKEN", "")
    
    # データベース設定
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/riskdb")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # セキュリティ
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-super-secret-key")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "jwt-secret-key")
    
    # API設定
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "5000"))
    
    # SSL設定
    SSL_CERT_PATH: str = os.getenv("SSL_CERT_PATH", "")
    SSL_KEY_PATH: str = os.getenv("SSL_KEY_PATH", "")
    
    # 監視
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    SENTRY_DSN: str = os.getenv("SENTRY_DSN", "")
    
    # パフォーマンス
    MAX_CONCURRENT_CALLS: int = int(os.getenv("MAX_CONCURRENT_CALLS", "100"))
    SCORING_TIMEOUT: int = int(os.getenv("SCORING_TIMEOUT", "5"))
    
    # 日本語特有設定
    LANGUAGE: str = "ja"
    TIMEZONE: str = "Asia/Tokyo"
    DATE_FORMAT: str = "%Y年%m月%d日"
    TIME_FORMAT: str = "%H時%M分"
```

### 2. **Kintone統合セットアップ**

```python
"""
本番環境Kintone統合
"""
import requests
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

class KintoneJapaneseProductionAPI:
    def __init__(self, subdomain: str, app_id: str, api_token: str):
        self.subdomain = subdomain
        self.app_id = app_id
        self.api_token = api_token
        self.base_url = f"https://{subdomain}.cybozu.com/k/v1"
        self.headers = {
            'X-Cybozu-API-Token': api_token,
            'Content-Type': 'application/json'
        }
        
    def get_customer_data(self, customer_id: str) -> Optional[Dict]:
        """Kintoneから顧客データを取得"""
        try:
            url = f"{self.base_url}/records.json"
            params = {
                'app': self.app_id,
                'query': f'顧客ID = "{customer_id}"',
                'fields': ['顧客ID', '未払FLAG', 'エージェント音量平均', 
                          '顧客感情スコア', '総会話時間', '顧客名', '電話番号', 
                          '口座残高', '支払履歴', 'リスク要因']
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data['records']:
                    return self.format_customer_data(data['records'][0])
            
            return None
            
        except Exception as e:
            logging.error(f"顧客データ取得エラー: {e}")
            return None
    
    def update_risk_score(self, customer_id: str, risk_data: Dict) -> bool:
        """Kintoneで顧客リスクスコアを更新"""
        try:
            url = f"{self.base_url}/record.json"
            
            data = {
                'app': self.app_id,
                'updateKey': {
                    'field': '顧客ID',
                    'value': customer_id
                },
                'record': {
                    'リスクスコア': {'value': risk_data['risk_score']},
                    'リスクレベル': {'value': risk_data['risk_level']},
                    'スコアリング時刻': {'value': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                    '主要リスク要因': {'value': ', '.join(risk_data.get('key_factors', []))},
                    'AI推奨事項': {'value': '\\n'.join(risk_data.get('recommendations', []))}
                }
            }
            
            response = requests.put(url, headers=self.headers, json=data)
            
            return response.status_code == 200
            
        except Exception as e:
            logging.error(f"リスクスコア更新エラー: {e}")
            return False
    
    def format_customer_data(self, record: Dict) -> Dict:
        """KintoneレコードデータをML処理用にフォーマット"""
        return {
            'customer_id': record.get('顧客ID', {}).get('value'),
            'customer_name': record.get('顧客名', {}).get('value'),
            'phone_number': record.get('電話番号', {}).get('value'),
            'payment_flag': record.get('未払FLAG', {}).get('value'),
            'agent_loudness_mean': float(record.get('エージェント音量平均', {}).get('value', 0)),
            'customer_sentiment_score': float(record.get('顧客感情スコア', {}).get('value', 0)),
            'total_conversation_duration': float(record.get('総会話時間', {}).get('value', 0)),
            'account_balance': float(record.get('口座残高', {}).get('value', 0)),
            'payment_history': record.get('支払履歴', {}).get('value', ''),
            'risk_factors': record.get('リスク要因', {}).get('value', ''),
            'historical_calls': self.get_customer_history(record.get('顧客ID', {}).get('value'))
        }
    
    def get_customer_history(self, customer_id: str) -> List[Dict]:
        """顧客の通話履歴を取得"""
        try:
            url = f"{self.base_url}/records.json"
            params = {
                'app': self.app_id,
                'query': f'顧客ID = "{customer_id}"',
                'orderBy': '開始タイムスタンプjst desc',
                'totalCount': True
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return [self.format_call_record(record) for record in data['records']]
            
            return []
            
        except Exception as e:
            logging.error(f"顧客履歴取得エラー: {e}")
            return []
    
    def format_call_record(self, record: Dict) -> Dict:
        """個別通話記録のフォーマット"""
        return {
            'timestamp': record.get('開始タイムスタンプjst', {}).get('value'),
            'agent_id': record.get('エージェント', {}).get('value'),
            'call_duration': record.get('総会話時間', {}).get('value'),
            'sentiment_score': record.get('顧客感情スコア', {}).get('value'),
            'payment_outcome': record.get('未払FLAG', {}).get('value')
        }
```

### 3. **本番環境対応APIサーバー**

```python
"""
本番環境対応リアルタイムAPIサーバー（日本語版）
"""
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import redis
import logging
import ssl
import os
from datetime import datetime
import jwt
from functools import wraps

from japanese_production_config import JapaneseProductionConfig
from kintone_japanese_production import KintoneJapaneseProductionAPI

# Flask アプリを本番設定で初期化
app = Flask(__name__)
app.config.from_object(JapaneseProductionConfig)

# セキュリティとCORS
CORS(app, origins=["https://yourdomain.com"])
socketio = SocketIO(app, cors_allowed_origins=["https://yourdomain.com"])

# レート制限
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# セッション管理用Redis
redis_client = redis.from_url(JapaneseProductionConfig.REDIS_URL)

# ログ設定
logging.basicConfig(
    level=getattr(logging, JapaneseProductionConfig.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 統合の初期化
kintone_api = KintoneJapaneseProductionAPI(
    subdomain=JapaneseProductionConfig.KINTONE_SUBDOMAIN,
    app_id=JapaneseProductionConfig.KINTONE_APP_ID,
    api_token=JapaneseProductionConfig.KINTONE_API_TOKEN
)

def require_auth(f):
    """認証デコレータ"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'トークンが提供されていません'}), 401
        
        try:
            token = token.replace('Bearer ', '')
            jwt.decode(token, JapaneseProductionConfig.JWT_SECRET, algorithms=['HS256'])
        except jwt.InvalidTokenError:
            return jsonify({'error': '無効なトークンです'}), 401
        
        return f(*args, **kwargs)
    
    return decorated

@app.route('/api/health')
def health_check():
    """ヘルスチェックエンドポイント"""
    return jsonify({
        'status': '正常',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'version': '1.0.0'
    })

@app.route('/api/score_call', methods=['POST'])
@limiter.limit("10 per minute")
@require_auth
def score_call():
    """本番環境通話スコアリングエンドポイント"""
    try:
        data = request.json
        call_id = data.get('call_id')
        customer_id = data.get('customer_id')
        
        # 入力の検証
        if not call_id or not customer_id:
            return jsonify({'error': '必須フィールドが不足しています'}), 400
        
        # Kintoneから顧客データを取得
        customer_data = kintone_api.get_customer_data(customer_id)
        
        if not customer_data:
            logger.warning(f"顧客が見つかりません: {customer_id}")
            return jsonify({'error': '顧客が見つかりません'}), 404
        
        # リアルタイムスコアリングの処理
        risk_score = process_real_time_scoring(data, customer_data)
        
        # Kintoneで新しいスコアを更新
        kintone_api.update_risk_score(customer_id, risk_score)
        
        # 結果をキャッシュ
        redis_client.setex(f"score:{call_id}", 3600, str(risk_score['risk_score']))
        
        # 接続されたクライアントに送信
        socketio.emit('risk_update', risk_score, room=f"call_{call_id}")
        
        logger.info(f"通話スコアリング完了 {call_id}: {risk_score['risk_score']}")
        
        return jsonify({'status': '成功', 'result': risk_score})
        
    except Exception as e:
        logger.error(f"通話スコアリングエラー: {e}")
        return jsonify({'error': '内部サーバーエラー'}), 500

def process_real_time_scoring(call_data, customer_data):
    """本番環境MLモデルでリアルタイムスコアリングを処理"""
    # 本番環境モデルの読み込み
    import joblib
    model = joblib.load('/app/models/production_risk_model.pkl')
    scaler = joblib.load('/app/models/production_scaler.pkl')
    
    # 特徴エンジニアリング
    features = engineer_features(call_data, customer_data)
    
    # 特徴のスケーリング
    features_scaled = scaler.transform([features])
    
    # リスク予測
    risk_probability = model.predict_proba(features_scaled)[0][1]
    risk_score = risk_probability * 100
    
    # リスクレベルの決定
    if risk_score >= 70:
        risk_level = "高リスク"
        recommendations = [
            "🚨 高リスク顧客 - 上司にエスカレーションしてください",
            "💰 支払いプランのオプションを提示してください",
            "📋 支払いに関する懸念を詳細に記録してください"
        ]
    elif risk_score >= 40:
        risk_level = "中リスク"
        recommendations = [
            "⚠️ 支払い約束を注意深く監視してください",
            "🤝 信頼関係を築いてください",
            "📞 フォローアップコールをスケジュールしてください"
        ]
    else:
        risk_level = "低リスク"
        recommendations = [
            "✅ 顧客は良好な支払い可能性を示しています",
            "🎯 アップセリング機会に焦点を当ててください",
            "😊 良好な関係を維持してください"
        ]
    
    return {
        'call_id': call_data['call_id'],
        'customer_id': call_data['customer_id'],
        'risk_score': round(risk_score, 1),
        'risk_level': risk_level,
        'confidence': round(max(risk_probability, 1-risk_probability), 3),
        'recommendations': recommendations,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def engineer_features(call_data, customer_data):
    """MLモデル用の特徴エンジニアリング"""
    return [
        call_data.get('agent_loudness_mean', 0),
        call_data.get('agent_sentiment_score', 0),
        call_data.get('customer_loudness_mean', 0),
        call_data.get('customer_sentiment_score', 0),
        call_data.get('total_conversation_duration', 0),
        customer_data.get('agent_loudness_mean', 0),
        customer_data.get('customer_sentiment_score', 0),
        len(customer_data.get('historical_calls', []))
    ]

# WebSocketイベント
@socketio.on('join_call')
def handle_join_call(data):
    """リアルタイム更新用の通話ルームに参加"""
    call_id = data.get('call_id')
    if call_id:
        join_room(f"call_{call_id}")
        emit('status', {'message': f'通話 {call_id} に参加しました'})

if __name__ == '__main__':
    # SSL付き本番環境サーバー
    context = None
    if JapaneseProductionConfig.SSL_CERT_PATH and JapaneseProductionConfig.SSL_KEY_PATH:
        context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        context.load_cert_chain(JapaneseProductionConfig.SSL_CERT_PATH, 
                               JapaneseProductionConfig.SSL_KEY_PATH)
    
    socketio.run(
        app,
        host=JapaneseProductionConfig.API_HOST,
        port=JapaneseProductionConfig.API_PORT,
        ssl_context=context,
        debug=False
    )
```

### 4. **Dockerデプロイメント**

```dockerfile
# Dockerfile - 本番環境日本語対応
FROM python:3.9-slim

# 作業ディレクトリの設定
WORKDIR /app

# 日本語ロケールのインストール
RUN apt-get update && apt-get install -y \
    locales \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && locale-gen ja_JP.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# 環境変数の設定
ENV LANG=ja_JP.UTF-8
ENV LANGUAGE=ja_JP:ja
ENV LC_ALL=ja_JP.UTF-8

# 要件のコピー
COPY requirements.txt .

# Python依存関係のインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードのコピー
COPY . .

# 非rootユーザーの作成
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# ポートの公開
EXPOSE 5000 8501 8502

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:5000/api/health || exit 1

# デフォルトコマンド
CMD ["python", "japanese_production_api_server.py"]
```

```yaml
# docker-compose.yml - 日本語本番環境
version: '3.8'

services:
  jp-api-server:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=redis://redis:6379
      - KINTONE_SUBDOMAIN=${KINTONE_SUBDOMAIN}
      - KINTONE_API_TOKEN=${KINTONE_API_TOKEN}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - CONNECT_INSTANCE_ID=${CONNECT_INSTANCE_ID}
      - LANGUAGE=ja
      - TIMEZONE=Asia/Tokyo    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  analytics-dashboard:
    build: .
    command: streamlit run streamlit_dashboard.py --server.port 8501 --server.address 0.0.0.0
    ports:
      - "8501:8501"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - LANGUAGE=ja
    depends_on:
      - postgres
    restart: unless-stopped
    
  mobile-dashboard:
    build: .
    command: streamlit run mobile_dashboard.py --server.port 8502 --server.address 0.0.0.0
    ports:
      - "8502:8502"
    environment:
      - LANGUAGE=ja
    restart: unless-stopped
    
  redis:
    image: redis:6-alpine
    restart: unless-stopped
    
  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=riskscoring
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - jp-api-server
      - streamlit-dashboard
      - mobile-dashboard
    restart: unless-stopped

volumes:
  postgres_data:
```

### 5. **本番環境デプロイメント手順**

```bash
#!/bin/bash
# 日本語対応本番環境デプロイスクリプト

echo "🚀 AI顧客リスクスコアリングシステムを本番環境にデプロイ中"

# 1. 環境変数の設定
export KINTONE_SUBDOMAIN="your-japanese-subdomain"
export KINTONE_APP_ID="your-app-id"
export KINTONE_API_TOKEN="your-api-token"
export CONNECT_INSTANCE_ID="your-connect-instance-id"
export AWS_ACCESS_KEY_ID="your-aws-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret"
export DATABASE_URL="postgresql://user:pass@localhost/riskdb"
export SECRET_KEY="your-super-secret-production-key"
export LANGUAGE="ja"
export TIMEZONE="Asia/Tokyo"

# 2. Dockerでビルドとデプロイ
echo "📦 Dockerコンテナをビルド中..."
docker-compose -f docker-compose.production.yml build

echo "🚀 本番環境サービスを開始中..."
docker-compose -f docker-compose.production.yml up -d

# 3. データベースマイグレーションの実行
echo "🗄️ データベースをセットアップ中..."
docker-compose exec jp-api-server python migrate_database.py

# 4. 本番環境MLモデルの読み込み
echo "🤖 MLモデルを読み込み中..."
docker-compose exec jp-api-server python load_production_model.py

# 5. デプロイメントの検証
echo "✅ デプロイメントを検証中..."
curl -f http://localhost:5000/api/health

echo "🎉 本番環境デプロイメント完了！"
echo "📊 分析ダッシュボード: https://yourdomain.com:8501"
echo "📱 モバイルダッシュボード: https://yourdomain.com:8502"
echo "🔗 APIサーバー: https://yourdomain.com:5000"
```

### 6. **設定チェックリスト**

本番環境用`.env`ファイルを作成:

```bash
# AWS設定
AWS_REGION=ap-northeast-1
CONNECT_INSTANCE_ID=your-connect-instance-id
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key

# Kintone設定
KINTONE_SUBDOMAIN=your-japanese-subdomain
KINTONE_APP_ID=your-app-id
KINTONE_API_TOKEN=your-api-token

# データベース
DATABASE_URL=postgresql://user:password@localhost:5432/riskscoring
REDIS_URL=redis://localhost:6379

# セキュリティ
SECRET_KEY=your-super-secret-production-key
JWT_SECRET=your-jwt-secret-key

# SSL
SSL_CERT_PATH=/path/to/your/cert.pem
SSL_KEY_PATH=/path/to/your/key.pem

# 監視
LOG_LEVEL=INFO
SENTRY_DSN=your-sentry-dsn

# 日本語設定
LANGUAGE=ja
TIMEZONE=Asia/Tokyo
DATE_FORMAT=%Y年%m月%d日
TIME_FORMAT=%H時%M分

# パフォーマンス
MAX_CONCURRENT_CALLS=100
SCORING_TIMEOUT=5
```

## 🎯 **次のステップ**

1. **AWS認証の取得** - AWSアカウントとConnectインスタンスの設定
2. **Kintoneの設定** - APIトークンとアプリIDの取得
3. **SSL証明書の設定** - 本番環境でのHTTPS
4. **クラウドへのデプロイ** - AWS ECS、Azure Container Instances、またはGoogle Cloud Run
5. **パフォーマンス監視の設定** - ログと監視の設定
6. **統合のテスト** - Amazon ConnectとKintone接続の検証

本番環境では、実際の営業通話中にリアルタイム顧客リスクスコアリングが利用可能になります！ 🚀

## 📞 **日本語カスタマーサポート**

- **技術サポート**: support@yourdomain.com
- **営業時間**: 平日 9:00-18:00 (JST)
- **緊急サポート**: +81-3-XXXX-XXXX
- **ドキュメント**: https://docs.yourdomain.com/ja

---

**最終更新**: 2025年5月30日  
**バージョン**: 1.0.0 日本語版  
**システムステータス**: ✅ 本番環境対応準備完了

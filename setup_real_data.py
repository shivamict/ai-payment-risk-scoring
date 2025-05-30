#!/usr/bin/env python3
"""
日本語データのセットアップスクリプト
"""

import os
import shutil
from pathlib import Path

def setup_real_data():
    """
    実際のExcelデータを使用するためのセットアップ手順
    """
    print("🚀 AI支払いリスク評価 - 実データセットアップ")
    print("=" * 50)
    
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data" / "raw"
    
    print(f"📁 データディレクトリ: {data_dir}")
    
    # データディレクトリの存在確認
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print("✅ データディレクトリを作成しました")
    
    # Excelファイルの検索
    excel_files = list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.xls"))
    
    if excel_files:
        print(f"📊 Excelファイルが見つかりました:")
        for file in excel_files:
            print(f"   - {file.name}")
        
        print("\n✅ Excelファイルを検出しました！システムは自動的にデータを使用します。")
        
    else:
        print("❌ データディレクトリにExcelファイルが見つかりません。")
        print("\n📋 実データを使用するには:")
        print(f"1. Excelファイルをこのディレクトリにコピーしてください: {data_dir}")
        print("2. 以下の列が含まれていることを確認してください:")
        
        expected_columns = [
            "未払FLAG", "レコード番号", "成約日date", "開始タイムスタンプjst", 
            "コンタクト_id", "エージェント", "営業担当者", "agent_loudness_mean",
            "agent_negative_sentence", "agent_neutral_sentence", "agent_positive_sentence",
            "agent_sentiment_score", "agent_talktime", "agent_talktime通話時間",
            "agent_total_sentence", "customer_loudness_mean", "customer_negative_sentence",
            "customer_neutral_sentence", "customer_positive_sentence", "customer_sentiment_score",
            "customer_talktime", "customer_talktime通話時間", "customer_total_sentence",
            "total_conversation_duration", "total_conversation_duration合計", "total_talktime",
            "total_talktime通話時間", "content_list", "電話日-成約日"
        ]
        
        for i, col in enumerate(expected_columns, 1):
            print(f"   {i:2d}. {col}")
        
        print(f"\n3. システムを実行: py src/main.py")
    
    print("\n🔧 設定:")
    print("- 対象列: 未払FLAG (支払いフラグ)")
    print("- 顧客ID: レコード番号 (レコード番号)")
    print("- 特徴量: すべてのエージェントと顧客のインタラクション指標")
    
    print("\n💡 ヒント:")
    print("- システムは顧客ごとの複数の通話を自動的に集計します")
    print("- 欠損値は自動的に処理されます")
    print("- 特徴量は通話インタラクションデータから生成されます")
    
    return len(excel_files) > 0

if __name__ == "__main__":
    has_data = setup_real_data()
    
    if has_data:
        print("\n🎉 実データで実行する準備ができました！")
        print("次のステップ: py src/main.py")
    else:
        print("\n⚠️ まずExcelファイルを追加してから、このスクリプトを再実行してください。")

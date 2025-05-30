#!/usr/bin/env python3
"""
Setup script to help configure the system for your real data
"""

import os
import shutil
from pathlib import Path

def setup_real_data():
    """
    Setup instructions for using your real Excel data
    """
    print("🚀 AI Payment Risk Scoring - Real Data Setup")
    print("=" * 50)
    
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data" / "raw"
    
    print(f"📁 Data directory: {data_dir}")
    
    # Check if data directory exists
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print("✅ Created data directory")
    
    # Look for Excel files
    excel_files = list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.xls"))
    
    if excel_files:
        print(f"📊 Found Excel files:")
        for file in excel_files:
            print(f"   - {file.name}")
        
        print("\n✅ Excel files detected! The system will automatically use your data.")
        
    else:
        print("❌ No Excel files found in the data directory.")
        print("\n📋 To use your real data:")
        print(f"1. Copy your Excel file to: {data_dir}")
        print("2. Make sure it contains these columns:")
        
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
        
        print(f"\n3. Run the system: py src/main.py")
    
    print("\n🔧 Configuration:")
    print("- Target column: 未払FLAG (payment flag)")
    print("- Customer ID: レコード番号 (record number)")
    print("- Features: All agent and customer interaction metrics")
    
    print("\n💡 Tips:")
    print("- The system will automatically aggregate multiple calls per customer")
    print("- Missing values will be handled automatically")
    print("- Features will be engineered from your call interaction data")
    
    return len(excel_files) > 0

if __name__ == "__main__":
    has_data = setup_real_data()
    
    if has_data:
        print("\n🎉 Ready to run with your real data!")
        print("Next step: py src/main.py")
    else:
        print("\n⚠️  Please add your Excel file first, then run this script again.")

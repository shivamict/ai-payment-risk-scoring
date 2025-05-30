#!/usr/bin/env python3
"""
Test script for AI Payment Risk Scoring System
This script tests the pipeline without requiring external dependencies
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test if all modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        import config
        print("✅ config.py imported successfully")
    except Exception as e:
        print(f"❌ Error importing config: {e}")
        return False
    
    # Test individual modules (they will fail on external imports but we can catch that)
    modules_to_test = [
        'data_preparation',
        'model_training', 
        'scoring',
        'utils',
        'main'
    ]
    
    for module_name in modules_to_test:
        try:
            module = __import__(module_name)
            print(f"✅ {module_name}.py structure is valid")
        except ImportError as e:
            if "numpy" in str(e) or "pandas" in str(e) or "sklearn" in str(e):
                print(f"⚠️  {module_name}.py structure is valid (missing dependencies: {str(e).split()[1]})")
            else:
                print(f"❌ {module_name}.py has structural issues: {e}")
                return False
        except Exception as e:
            print(f"❌ {module_name}.py has issues: {e}")
            return False
    
    return True

def test_file_structure():
    """Test if all required files and directories exist."""
    print("\n📁 Testing file structure...")
    
    base_path = Path(__file__).parent
    
    required_files = [
        "config.py",
        "requirements.txt", 
        "README.md",
        "streamlit_dashboard.py",
        "src/main.py",
        "src/data_preparation.py",
        "src/model_training.py",
        "src/scoring.py",
        "src/utils.py",
        "notebooks/exploratory_analysis.ipynb"
    ]
    
    required_dirs = [
        "src",
        "data",
        "data/raw",
        "models", 
        "outputs",
        "notebooks"
    ]
    
    all_exist = True
    
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            all_exist = False
    
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if full_path.exists() and full_path.is_dir():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ Missing directory: {dir_path}/")
            all_exist = False
    
    return all_exist

def test_config():
    """Test configuration settings."""
    print("\n⚙️  Testing configuration...")
    
    try:
        import config
        
        # Check if key configuration elements exist
        required_configs = [
            'DATA_DIR',
            'MODELS_DIR', 
            'OUTPUT_DIR',
            'MODEL_PARAMS',
            'RISK_THRESHOLDS'
        ]
        
        for config_name in required_configs:
            if hasattr(config, config_name):
                print(f"✅ {config_name}: {getattr(config, config_name)}")
            else:
                print(f"❌ Missing configuration: {config_name}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 AI Payment Risk Scoring System - Test Suite")
    print("=" * 50)
    
    # Run tests
    structure_ok = test_file_structure()
    config_ok = test_config() 
    imports_ok = test_imports()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"File Structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    print(f"Configuration: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"Module Structure: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    
    if structure_ok and config_ok and imports_ok:
        print("\n🎉 All tests passed! System is ready for dependency installation.")
        print("\n📋 Next Steps:")
        print("1. Install Python dependencies:")
        print("   py -m pip install -r requirements.txt")
        print("2. Run the complete pipeline:")
        print("   py src/main.py")
        print("3. Launch the dashboard:")
        print("   py -m streamlit run streamlit_dashboard.py")
        
        return True
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before proceeding.")
        return False

if __name__ == "__main__":
    main()

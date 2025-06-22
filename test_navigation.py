#!/usr/bin/env python3
"""
Test script to verify navigation and imports work correctly
"""

import sys
import os

# Add the ui directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ui'))

def test_imports():
    """Test if all required modules can be imported"""
    try:
        from pages.heart_disease_detection import show_heart_disease_detection
        print("✅ heart_disease_detection imported successfully")
    except Exception as e:
        print(f"❌ Error importing heart_disease_detection: {e}")
    
    try:
        from pages.healthcare_trends import show_healthcare_trends
        print("✅ healthcare_trends imported successfully")
    except Exception as e:
        print(f"❌ Error importing healthcare_trends: {e}")
    
    try:
        from pages.exploratory_analysis import show_exploratory_analysis
        print("✅ exploratory_analysis imported successfully")
    except Exception as e:
        print(f"❌ Error importing exploratory_analysis: {e}")

def test_files_exist():
    """Test if required files exist"""
    required_files = [
        "data/processed.cleveland.data",
        "models/svm_heart_model.pkl"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} not found")

if __name__ == "__main__":
    print("Testing navigation and imports...")
    print("=" * 50)
    test_imports()
    print("-" * 50)
    test_files_exist()
    print("=" * 50)
    print("Test completed!") 
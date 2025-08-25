#!/usr/bin/env python3
"""
Test script to verify the setup works correctly.
"""

import yaml
from pathlib import Path

def test_config_loading():
    """Test that configuration files can be loaded."""
    config_path = Path("configs/experiment.yaml")
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Successfully loaded config: {config_path}")
        print(f"   Dataset: {config['dataset']} ({config['dataset_language']})")
        print(f"   Prompt: {config['prompt']} ({config['prompt_language']})")
        print(f"   Model: {config['model']}")
        return True
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

def test_directory_structure():
    """Test that all required directories exist."""
    required_dirs = ["configs", "authorship_verification", "data", "results", "notebooks"]
    
    all_good = True
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ Directory exists: {dir_name}")
        else:
            print(f"❌ Directory missing: {dir_name}")
            all_good = False
    
    return all_good

def test_main_script():
    """Test that the main script can be imported."""
    try:
        import sys
        sys.path.append('authorship_verification')
        from authorship_verification import AuthorshipVerificationExperiment
        print("✅ Main script can be imported successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing main script: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing AV Setup")
    print("=" * 50)
    
    tests = [
        ("Config Loading", test_config_loading),
        ("Directory Structure", test_directory_structure), 
        ("Main Script Import", test_main_script)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}:")
        if test_func():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Setup is ready.")
        print("\n🚀 Next steps:")
        print("1. Copy .env.example to .env and add your API key")
        print("2. Run: python authorship_verification/authorship_verification.py --config configs/experiment.yaml")
    else:
        print("❌ Some tests failed. Please check the setup.")

if __name__ == "__main__":
    main()

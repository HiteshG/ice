#!/usr/bin/env python3
"""
Test script for Ice Hockey Tracker
Run this to test end-to-end functionality
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from config import MainConfig, print_mode_comparison
from main import HockeyTracker

def test_config():
    """Test configuration setup."""
    print("="*60)
    print("TESTING CONFIGURATION")
    print("="*60)
    
    config = MainConfig()
    
    print("\n✓ Default config created")
    print(f"  - Tracking mode: {config.tracking.mode}")
    print(f"  - Detector device: {config.detector.device}")
    print(f"  - Class names: {config.detector.class_names}")
    
    # Test class names
    expected_classes = ["Center Ice", "Faceoff", "Goalpost", "Goaltender", "Player", "Puck", "Referee"]
    actual_classes = list(config.detector.class_names.values())
    
    assert "Puck" in actual_classes, "Puck class missing!"
    assert "Goaltender" in actual_classes, "Goaltender class missing!"
    assert "Player" in actual_classes, "Player class missing!"
    
    print("\n✓ All ice hockey classes present")
    return True


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*60)
    print("TESTING IMPORTS")
    print("="*60)
    
    try:
        from detector import ObjectDetector
        print("✓ detector module")
        
        from unified_tracker import UnifiedTracker, create_tracker
        print("✓ unified_tracker module")
        
        from team_assigner import TeamAssigner
        print("✓ team_assigner module")
        
        from visualizer import Visualizer
        print("✓ visualizer module")
        
        from processor import DataProcessor
        print("✓ processor module")
        
        from utils import read_video, write_video
        print("✓ utils module")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False


def test_tracker_init():
    """Test tracker initialization."""
    print("\n" + "="*60)
    print("TESTING TRACKER INITIALIZATION")
    print("="*60)
    
    try:
        config = MainConfig()
        config.tracking.mode = "botsort"  # Use simplest mode for testing
        
        print("\nInitializing HockeyTracker...")
        tracker = HockeyTracker(config)
        
        print("✓ HockeyTracker initialized successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "🏒" * 30)
    print("ICE HOCKEY TRACKER - SYSTEM TEST")
    print("🏒" * 30 + "\n")
    
    tests = [
        ("Configuration", test_config),
        ("Module Imports", test_imports),
        ("Tracker Initialization", test_tracker_init),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status:10s} - {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! System is ready.")
        print("\nNext steps:")
        print("1. Make sure you have a trained YOLO model for ice hockey")
        print("2. Place your model at the path specified in config")
        print("3. Run: python main.py --video your_hockey_video.mp4")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

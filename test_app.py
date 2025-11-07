#!/usr/bin/env python3
"""
Test script for the Financial Project Application
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported successfully."""
    try:
        print("Testing imports...")
        
        # Test main module
        import main
        print("✓ Main module imported successfully")
        
        # Test utility modules
        from utils.logger import setup_logger
        from utils.data_fetcher import DataFetcher
        print("✓ Utility modules imported successfully")
        
        # Test application modules
        from modules.dl_module import DeepLearningModule
        from modules.nlp_module import NLPModule
        from modules.is2_module import IS2Module
        print("✓ Application modules imported successfully")
        
        print("\nAll imports successful! The application is ready to run.")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_data_fetcher():
    """Test the data fetcher with a simple example."""
    try:
        print("\nTesting data fetcher...")
        
        from utils.data_fetcher import DataFetcher
        
        # Initialize data fetcher
        fetcher = DataFetcher()
        print("✓ DataFetcher initialized successfully")
        
        # Test with a simple stock (this will use Yahoo Finance fallback)
        print("Testing data fetch (this may take a moment)...")
        data = fetcher.fetch_stock_data("AAPL", "2023-01-01", "2023-12-31")
        
        if not data.empty:
            print(f"✓ Successfully fetched {len(data)} records for AAPL")
            print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
        else:
            print("✗ No data returned")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Data fetcher test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*50)
    print("FINANCIAL PROJECT APPLICATION - TEST SUITE")
    print("="*50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check your Python environment.")
        return False
    
    # Test data fetcher
    if not test_data_fetcher():
        print("\n❌ Data fetcher tests failed. Please check your internet connection.")
        return False
    
    print("\n" + "="*50)
    print("✅ ALL TESTS PASSED!")
    print("The application is ready to use.")
    print("\nTo run the application:")
    print("  python main.py")
    print("\nTo run specific modules:")
    print("  python main.py --module dl")
    print("  python main.py --module nlp")
    print("  python main.py --module is2")
    print("="*50)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)




#!/usr/bin/env python3
"""
Setup script for the Financial Project Application
"""

import os
import sys
import subprocess

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("Creating necessary directories...")
    directories = ["logs", "models", "data"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory already exists: {directory}")
    
    return True

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    if sys.version_info < (3, 7):
        print("✗ Python 3.7 or higher is required")
        return False
    else:
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        return True

def main():
    """Main setup function."""
    print("="*60)
    print("FINANCIAL PROJECT APPLICATION - SETUP")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. (Optional) Set your Alpha Vantage API key:")
    print("   export ALPHA_VANTAGE_API_KEY='your_api_key_here'")
    print("\n2. Run the test suite:")
    print("   python test_app.py")
    print("\n3. Start the application:")
    print("   python main.py")
    print("\n4. Or run specific modules:")
    print("   python main.py --module dl")
    print("   python main.py --module nlp")
    print("   python main.py --module is2")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

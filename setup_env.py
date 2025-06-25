#!/usr/bin/env python3
"""
Environment setup script for CompI project.
Run this script to check and install dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please use Python 3.8 or higher")
        return False

def check_gpu():
    """Check for CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available with {gpu_count} GPU(s): {gpu_name}")
            return True
        else:
            print("⚠️  CUDA not available, will use CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet, GPU check will be done after installation")
        return False

def install_requirements():
    """Install requirements from requirements.txt."""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing requirements"
    )

def download_nltk_data():
    """Download required NLTK data."""
    try:
        import nltk
        print("\n🔄 Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded")
        return True
    except ImportError:
        print("⚠️  NLTK not installed, skipping data download")
        return False

def setup_textblob():
    """Setup TextBlob corpora."""
    try:
        import textblob
        print("\n🔄 Setting up TextBlob...")
        run_command(f"{sys.executable} -m textblob.download_corpora", 
                   "Downloading TextBlob corpora")
        return True
    except ImportError:
        print("⚠️  TextBlob not installed, skipping setup")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up CompI Development Environment")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Check GPU after PyTorch installation
    check_gpu()
    
    # Setup additional components
    download_nltk_data()
    setup_textblob()
    
    print("\n" + "=" * 50)
    print("🎉 Environment setup completed!")
    print("\nNext steps:")
    print("1. Run: python src/test_setup.py")
    print("2. Start experimenting with notebooks/")
    print("3. Check out the README.md for usage examples")

if __name__ == "__main__":
    main()

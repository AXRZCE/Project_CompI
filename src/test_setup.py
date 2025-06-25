#!/usr/bin/env python3
"""
Test script to verify CompI environment setup.
Run this after installing dependencies to ensure everything works.
"""

import sys
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def test_basic_imports():
    """Test basic Python library imports."""
    print("🔍 Testing basic imports...")
    
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        print("✅ Basic scientific libraries imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Basic import failed: {e}")
        return False

def test_pytorch():
    """Test PyTorch installation and CUDA availability."""
    print("🔍 Testing PyTorch...")
    
    try:
        import torch
        import torchvision
        import torchaudio
        
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        
        # Test CUDA
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {device_count} GPU(s) - {device_name}")
        else:
            print("⚠️  CUDA not available, using CPU")
        
        # Test basic tensor operations
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.matmul(x, y)
        print("✅ Basic tensor operations working")
        
        return True
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_transformers():
    """Test Hugging Face Transformers."""
    print("🔍 Testing Transformers...")
    
    try:
        from transformers import pipeline
        
        # Test a simple sentiment analysis pipeline
        classifier = pipeline("sentiment-analysis", 
                             model="distilbert-base-uncased-finetuned-sst-2-english")
        result = classifier("This is a test sentence.")
        print(f"✅ Transformers working - Sentiment: {result[0]['label']}")
        
        return True
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Transformers test failed: {e}")
        return False

def test_diffusers():
    """Test Diffusers library."""
    print("🔍 Testing Diffusers...")
    
    try:
        from diffusers import StableDiffusionPipeline
        print("✅ Diffusers imported successfully")
        
        # Note: We don't load the full model here to save time and memory
        # Just test that the import works
        return True
    except ImportError as e:
        print(f"❌ Diffusers import failed: {e}")
        return False

def test_audio_libraries():
    """Test audio processing libraries."""
    print("🔍 Testing audio libraries...")
    
    try:
        import librosa
        import soundfile as sf
        
        # Test basic audio functionality
        sr = 22050
        duration = 1.0
        t = librosa.frames_to_time(range(int(sr * duration)), sr=sr)
        print("✅ Audio libraries working")
        
        return True
    except ImportError as e:
        print(f"❌ Audio libraries import failed: {e}")
        return False

def test_nlp_libraries():
    """Test NLP libraries."""
    print("🔍 Testing NLP libraries...")
    
    try:
        import textblob
        from textblob import TextBlob
        
        # Test basic NLP functionality
        blob = TextBlob("This is a test sentence.")
        sentiment = blob.sentiment
        print(f"✅ TextBlob working - Polarity: {sentiment.polarity:.2f}")
        
        return True
    except ImportError as e:
        print(f"❌ NLP libraries import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️  NLP test warning: {e}")
        return True  # Continue even if NLTK data is missing

def test_ui_libraries():
    """Test UI libraries."""
    print("🔍 Testing UI libraries...")
    
    try:
        import streamlit
        import gradio
        print("✅ UI libraries imported successfully")
        return True
    except ImportError as e:
        print(f"❌ UI libraries import failed: {e}")
        return False

def test_project_structure():
    """Test project structure and imports."""
    print("🔍 Testing project structure...")
    
    try:
        # Test project imports
        sys.path.append(str(Path(__file__).parent.parent))
        
        from src.config import PROJECT_ROOT, DATA_DIR, OUTPUTS_DIR
        from src.utils.logging_utils import setup_logger
        
        # Test logger
        logger = setup_logger("test")
        logger.info("Test log message")
        
        print("✅ Project structure and imports working")
        return True
    except ImportError as e:
        print(f"❌ Project import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Project structure test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 CompI Environment Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_pytorch,
        test_transformers,
        test_diffusers,
        test_audio_libraries,
        test_nlp_libraries,
        test_ui_libraries,
        test_project_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Empty line for readability
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your environment is ready.")
        print("\nNext steps:")
        print("1. Check out notebooks/ for examples")
        print("2. Start with a simple text-to-image generation")
        print("3. Explore multi-modal combinations")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("You may need to install missing dependencies or fix configuration.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

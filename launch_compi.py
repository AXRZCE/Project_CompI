#!/usr/bin/env python3
"""
CompI Launcher Script

Simple launcher for the CompI application that handles encoding issues
and provides better error reporting.
"""

import os
import sys
import subprocess

def main():
    """Launch the CompI Streamlit application"""
    
    print("🚀 Launching CompI - Complete Multimodal AI Art Platform")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("compi_complete_app.py"):
        print("❌ Error: compi_complete_app.py not found!")
        print("Please run this script from the CompI project directory.")
        return 1
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required!")
        return 1
    
    print(f"✅ Python version: {sys.version}")
    
    # Check if Streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Error: Streamlit is not installed!")
        print("Please install it with: pip install streamlit")
        return 1
    
    # Check core dependencies
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ Device: {device.upper()}")
        
        if device == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✅ GPU: {gpu_name}")
            except:
                print("✅ CUDA available but no GPU name detected")
                
    except ImportError:
        print("❌ Error: PyTorch is not installed!")
        return 1
    
    print("\n🌐 Starting Streamlit server...")
    print("📱 The app will open in your default browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Launch Streamlit with proper encoding
    try:
        # Set environment variables for proper encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "compi_complete_app.py",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd, env=env, check=True)
        
    except KeyboardInterrupt:
        print("\n\n👋 CompI application stopped by user")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error launching Streamlit: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

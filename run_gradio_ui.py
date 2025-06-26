#!/usr/bin/env python3
"""
Convenience script to run the CompI Gradio UI.
This script launches the interactive web interface for CompI using Gradio.
"""

import os
import sys
import subprocess

def main():
    # Get the directory of this script (project root)
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the Gradio UI script
    ui_script = os.path.join(project_root, "src", "ui", "compi_phase1c_gradio_ui.py")
    
    # Check if gradio is available
    try:
        import gradio
    except ImportError:
        print("❌ Gradio not found. Please install it:")
        print("   pip install gradio")
        sys.exit(1)
    
    # Run Gradio UI
    print("🚀 Starting CompI Interactive UI (Gradio)...")
    print("📱 The web interface will open in your browser automatically")
    print("🔗 If it doesn't open, go to: http://localhost:7860")
    print("⏹️  Press Ctrl+C to stop the server")
    print()
    
    try:
        cmd = [sys.executable, ui_script]
        subprocess.run(cmd, cwd=project_root)
    except KeyboardInterrupt:
        print("\n👋 CompI UI stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error running UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convenience script to run advanced text-to-image generation.
This script calls the organized generator in src/generators/
"""

import os
import sys
import subprocess

def main():
    # Get the directory of this script (project root)
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the actual generator script
    generator_script = os.path.join(project_root, "src", "generators", "compi_phase1_advanced.py")
    
    # Pass all command line arguments to the generator
    cmd = [sys.executable, generator_script] + sys.argv[1:]
    
    # Run the generator
    try:
        subprocess.run(cmd, check=True, cwd=project_root)
    except subprocess.CalledProcessError as e:
        print(f"Error running generator: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

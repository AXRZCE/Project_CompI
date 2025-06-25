"""
Configuration settings for CompI project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUTS_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model configurations
STABLE_DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Generation settings
DEFAULT_IMAGE_SIZE = (512, 512)
DEFAULT_INFERENCE_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 7.5

# Audio settings
SAMPLE_RATE = 22050
AUDIO_DURATION = 10  # seconds

# Device settings
DEVICE = "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"

# API keys (load from environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

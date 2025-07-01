"""
File handling utilities for CompI project.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Union
from PIL import Image
import soundfile as sf
import numpy as np

from src.config import OUTPUTS_DIR

def save_image(image: Image.Image, filename: str, subfolder: str = "images") -> Path:
    """
    Save a PIL Image to the outputs directory.
    
    Args:
        image: PIL Image to save
        filename: Name of the file (with extension)
        subfolder: Subfolder within outputs directory
        
    Returns:
        Path to saved file
    """
    output_dir = OUTPUTS_DIR / subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = output_dir / filename
    image.save(file_path)
    
    return file_path

def save_audio(audio_data: np.ndarray, filename: str, 
               sample_rate: int = 22050, subfolder: str = "audio") -> Path:
    """
    Save audio data to the outputs directory.
    
    Args:
        audio_data: Audio data as numpy array
        filename: Name of the file (with extension)
        sample_rate: Audio sample rate
        subfolder: Subfolder within outputs directory
        
    Returns:
        Path to saved file
    """
    output_dir = OUTPUTS_DIR / subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = output_dir / filename
    sf.write(file_path, audio_data, sample_rate)
    
    return file_path

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yml', '.yaml']:
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """
    Alias for ensure_dir for backward compatibility.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    return ensure_dir(path)

def generate_filename(prompt: str, style: str = "", mood: str = "",
                     seed: int = 0, variation: int = 1,
                     has_audio: bool = False, max_length: int = 100) -> str:
    """
    Generate a descriptive filename for generated images.

    Args:
        prompt: Text prompt used for generation
        style: Art style
        mood: Mood/atmosphere
        seed: Random seed used
        variation: Variation number
        has_audio: Whether audio was used in generation
        max_length: Maximum filename length

    Returns:
        Generated filename (without extension)
    """
    import re
    from datetime import datetime

    # Clean and truncate prompt
    prompt_clean = re.sub(r'[^\w\s-]', '', prompt.lower())
    prompt_slug = "_".join(prompt_clean.split()[:6])[:30]

    # Clean style and mood
    style_slug = re.sub(r'[^\w]', '', style.lower())[:10] if style else ""
    mood_slug = re.sub(r'[^\w]', '', mood.lower())[:10] if mood else ""

    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Audio indicator
    audio_tag = "_audio" if has_audio else ""

    # Combine all parts
    parts = [prompt_slug, style_slug, mood_slug, timestamp, f"seed{seed}", f"v{variation}"]
    filename = "_".join(filter(None, parts)) + audio_tag

    # Truncate if too long
    if len(filename) > max_length:
        filename = filename[:max_length]

    return filename

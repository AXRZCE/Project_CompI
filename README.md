# CompI - Compositional Intelligence Project

A multi-modal AI system that generates creative content by combining text, images, audio, and emotional context.

## 🚀 Project Overview

CompI (Compositional Intelligence) is designed to create rich, contextually-aware content by:

- Processing text prompts with emotional analysis
- Generating images using Stable Diffusion
- Creating audio compositions
- Combining multiple modalities for enhanced creative output

## 📁 Project Structure

```
Project CompI/
├── src/                    # Source code
│   ├── generators/        # Image generation modules
│   ├── models/            # Model implementations
│   ├── utils/             # Utility functions
│   ├── data/              # Data processing
│   ├── ui/                # User interface components
│   └── setup_env.py       # Environment setup script
├── notebooks/             # Jupyter notebooks for experimentation
├── data/                  # Dataset storage
├── outputs/               # Generated content
├── tests/                 # Unit tests
├── run_*.py               # Convenience scripts for generators
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🛠️ Setup Instructions

### 1. Create Virtual Environment

```bash
# Using conda (recommended for ML projects)
conda create -n compi-env python=3.10 -y
conda activate compi-env

# OR using venv
python -m venv compi-env
# Windows
compi-env\Scripts\activate
# Linux/Mac
source compi-env/bin/activate
```

### 2. Install Dependencies

**For GPU users (recommended for faster generation):**

```bash
# First, check your CUDA version
nvidia-smi

# Install PyTorch with CUDA support first (replace cu121 with your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install remaining requirements
pip install -r requirements.txt
```

**For CPU-only users:**

```bash
pip install -r requirements.txt
```

### 3. Test Installation

```bash
python src/test_setup.py
```

## 🚀 Quick Start

```bash
# Basic text-to-image generation
python run_basic_generation.py "A magical forest, digital art"

# Advanced generation with style conditioning
python run_advanced_styling.py "dragon in a crystal cave" --style "oil painting" --mood "dramatic"

# Interactive style selection
python run_styled_generation.py
```

## 🎯 Core Features

- **Text Analysis**: Emotion detection and sentiment analysis
- **Image Generation**: Stable Diffusion integration
- **Audio Processing**: Music and sound generation
- **Multi-modal Fusion**: Combining different content types

## 🔧 Tech Stack

- **Deep Learning**: PyTorch, Transformers, Diffusers
- **Audio**: librosa, soundfile
- **UI**: Streamlit/Gradio
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn

## 📝 Usage

Coming soon - basic usage examples and API documentation.

## 🤝 Contributing

This is a development project. Feel free to experiment and extend functionality.

## 📄 License

MIT License - see LICENSE file for details.

# Project_CompI

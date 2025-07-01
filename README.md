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

### Phase 1: Text-to-Image Generation

```bash
# Basic text-to-image generation
python run_basic_generation.py "A magical forest, digital art"

# Advanced generation with style conditioning
python run_advanced_styling.py "dragon in a crystal cave" --style "oil painting" --mood "dramatic"

# Interactive style selection
python run_styled_generation.py

# Quality evaluation and analysis
python run_evaluation.py

# Personal style training with LoRA
python run_lora_training.py --dataset-dir datasets/my_style

# Generate with personal style
python run_style_generation.py --lora-path lora_models/my_style/checkpoint-1000 "artwork in my_style"
```

### Phase 2.A: Audio-to-Image Generation 🎵

```bash
# Install audio processing dependencies
pip install openai-whisper

# Streamlit UI (Recommended)
streamlit run src/ui/compi_phase2a_streamlit_ui.py

# Command line generation
python run_phase2a_audio_to_image.py --prompt "mystical forest" --audio "music.mp3"

# Interactive mode
python run_phase2a_audio_to_image.py --interactive

# Test installation
python src/test_phase2a.py

# Run examples
python examples/phase2a_audio_examples.py --example all
```

### Phase 2.B: Data/Logic-to-Image Generation 📊

```bash
# Streamlit UI (Recommended)
streamlit run src/ui/compi_phase2b_streamlit_ui.py

# Command line generation with CSV data
python run_phase2b_data_to_image.py --prompt "data visualization" --csv "data.csv"

# Mathematical formula generation
python run_phase2b_data_to_image.py --prompt "mathematical harmony" --formula "np.sin(np.linspace(0, 4*np.pi, 100))"

# Batch processing
python run_phase2b_data_to_image.py --batch-csv "data_folder/" --prompt "scientific patterns"

# Interactive mode
python run_phase2b_data_to_image.py --interactive
```

### Phase 2.C: Emotional/Contextual Input to Image Generation 🌀

```bash
# Streamlit UI (Recommended)
streamlit run src/ui/compi_phase2c_streamlit_ui.py

# Command line generation with preset emotion
python run_phase2c_emotion_to_image.py --prompt "mystical forest" --emotion "mysterious"

# Custom emotion generation
python run_phase2c_emotion_to_image.py --prompt "urban landscape" --emotion "🤩" --type custom

# Descriptive emotion generation
python run_phase2c_emotion_to_image.py --prompt "mountain vista" --emotion "I feel a sense of wonder" --type text

# Batch emotion processing
python run_phase2c_emotion_to_image.py --batch-emotions "joyful,sad,mysterious" --prompt "abstract art"

# Interactive mode
python run_phase2c_emotion_to_image.py --interactive
```

### Phase 2.D: Real-Time Data Feeds to Image Generation 🌎

```bash
# Streamlit UI (Recommended)
streamlit run src/ui/compi_phase2d_streamlit_ui.py

# Command line generation with weather data
python run_phase2d_realtime_to_image.py --prompt "cityscape" --weather --city "Tokyo"

# News-driven generation
python run_phase2d_realtime_to_image.py --prompt "abstract art" --news --category "technology"

# Multi-source generation
python run_phase2d_realtime_to_image.py --prompt "world state" --weather --news --financial

# Temporal series generation
python run_phase2d_realtime_to_image.py --prompt "evolving world" --weather --temporal "0,30,60"

# Interactive mode
python run_phase2d_realtime_to_image.py --interactive
```

## 🎯 Core Features

- **Text Analysis**: Emotion detection and sentiment analysis
- **Image Generation**: Stable Diffusion integration with advanced conditioning
- **Audio Processing**: Music and sound analysis with Whisper integration
- **Data Processing**: CSV analysis and mathematical formula evaluation
- **Emotion Processing**: Preset emotions, custom emotions, emoji, and contextual analysis
- **Real-Time Integration**: Live weather, news, and financial data feeds
- **Multi-modal Fusion**: Combining text, audio, data, emotions, and real-time feeds
- **Pattern Recognition**: Automatic detection of trends, correlations, and seasonality
- **Poetic Interpretation**: Converting data patterns and emotions into artistic language
- **Color Psychology**: Emotion-based color palette generation and conditioning
- **Temporal Awareness**: Time-sensitive data processing and evolution tracking

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

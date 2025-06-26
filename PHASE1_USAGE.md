# CompI Phase 1: Text-to-Image Generation Usage Guide

This guide covers the Phase 1 implementation of CompI's text-to-image generation capabilities using Stable Diffusion.

## 🚀 Quick Start

### Basic Usage

```bash
# Simple generation with interactive prompt
python run_basic_generation.py

# Generate from command line
python run_basic_generation.py "A magical forest, digital art, highly detailed"

# Or run directly from src/generators/
python src/generators/compi_phase1_text2image.py "A magical forest"
```

### Advanced Usage

```bash
# Advanced script with more options
python run_advanced_generation.py "cyberpunk city at sunset" --negative "blurry, low quality" --steps 50 --batch 3

# Interactive mode for experimentation
python run_advanced_generation.py --interactive

# Or run directly from src/generators/
python src/generators/compi_phase1_advanced.py --interactive
```

## 📋 Available Scripts

### 1. `compi_phase1_text2image.py` - Basic Implementation

**Features:**

- Simple, standalone text-to-image generation
- Automatic GPU/CPU detection
- Command line or interactive prompts
- Automatic output saving with descriptive filenames
- Comprehensive logging

**Usage:**

```bash
python compi_phase1_text2image.py [prompt]
```

### 2. `compi_phase1_advanced.py` - Enhanced Implementation

**Features:**

- Batch generation (multiple images)
- Negative prompts (what to avoid)
- Customizable parameters (steps, guidance, dimensions)
- Interactive mode for experimentation
- Metadata saving (JSON files with generation parameters)
- Multiple model support

**Command Line Options:**

```bash
python compi_phase1_advanced.py [OPTIONS] [PROMPT]

Options:
  --negative, -n TEXT     Negative prompt (what to avoid)
  --steps, -s INTEGER     Number of inference steps (default: 30)
  --guidance, -g FLOAT    Guidance scale (default: 7.5)
  --seed INTEGER          Random seed for reproducibility
  --batch, -b INTEGER     Number of images to generate
  --width, -w INTEGER     Image width (default: 512)
  --height INTEGER        Image height (default: 512)
  --model, -m TEXT        Model to use (default: runwayml/stable-diffusion-v1-5)
  --output, -o TEXT       Output directory (default: outputs)
  --interactive, -i       Interactive mode
```

## 🎨 Example Commands

### Basic Examples

```bash
# Simple landscape
python run_basic_generation.py "serene mountain lake, golden hour, photorealistic"

# Digital art style
python run_basic_generation.py "futuristic robot, neon lights, cyberpunk style, digital art"
```

### Advanced Examples

```bash
# High-quality generation with negative prompts
python run_advanced_generation.py "beautiful portrait of a woman, oil painting style" \
  --negative "blurry, distorted, low quality, bad anatomy" \
  --steps 50 --guidance 8.0

# Batch generation with fixed seed
python run_advanced_generation.py "abstract geometric patterns, colorful" \
  --batch 5 --seed 12345 --steps 40

# Custom dimensions for landscape
python run_advanced_generation.py "panoramic view of alien landscape" \
  --width 768 --height 512 --steps 35

# Interactive experimentation
python run_advanced_generation.py --interactive
```

## 📁 Output Structure

Generated images are saved in the `outputs/` directory with descriptive filenames:

```
outputs/
├── magical_forest_digital_art_20241225_143022_seed42.png
├── magical_forest_digital_art_20241225_143022_seed42_metadata.json
├── cyberpunk_city_sunset_20241225_143156_seed1337.png
└── cyberpunk_city_sunset_20241225_143156_seed1337_metadata.json
```

### Metadata Files

Each generated image (in advanced mode) includes a JSON metadata file with:

- Original prompt and negative prompt
- Generation parameters (steps, guidance, seed)
- Image dimensions and model used
- Timestamp and batch information

## ⚙️ Configuration Tips

### For Best Quality

- Use 30-50 inference steps
- Guidance scale 7.5-12.0
- Include style descriptors ("digital art", "oil painting", "photorealistic")
- Use negative prompts to avoid unwanted elements

### For Speed

- Use 20-25 inference steps
- Lower guidance scale (6.0-7.5)
- Stick to 512x512 resolution

### For Experimentation

- Use interactive mode
- Try different seeds with the same prompt
- Experiment with guidance scale values
- Use batch generation to explore variations

## 🔧 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or image dimensions
2. **Slow generation**: Ensure CUDA is available and working
3. **Poor quality**: Increase steps, adjust guidance scale, improve prompts
4. **Model download fails**: Check internet connection, try again

### Performance Optimization

- The scripts automatically enable attention slicing for memory efficiency
- GPU detection is automatic
- Models are cached after first download

## 🎨 Phase 1.B: Style Conditioning & Prompt Engineering

### 3. `compi_phase1b_styled_generation.py` - Style Conditioning

**Features:**

- Interactive style and mood selection from curated lists
- Intelligent prompt engineering and combination
- Multiple variations with unique seeds
- Comprehensive logging and filename organization

**Usage:**

```bash
python run_styled_generation.py [prompt]
# Or directly: python src/generators/compi_phase1b_styled_generation.py [prompt]
```

### 4. `compi_phase1b_advanced_styling.py` - Advanced Style Control

**Features:**

- 13 predefined art styles with optimized prompts and negative prompts
- 9 mood categories with atmospheric conditioning
- Quality presets (draft/standard/high)
- Command line and interactive modes
- Comprehensive metadata saving

**Command Line Options:**

```bash
python run_advanced_styling.py [OPTIONS] [PROMPT]
# Or directly: python src/generators/compi_phase1b_advanced_styling.py [OPTIONS] [PROMPT]

Options:
  --style, -s TEXT        Art style (or number from list)
  --mood, -m TEXT         Mood/atmosphere (or number from list)
  --variations, -v INT    Number of variations (default: 1)
  --quality, -q CHOICE    Quality preset [draft/standard/high]
  --negative, -n TEXT     Negative prompt
  --interactive, -i       Interactive mode
  --list-styles          List available styles and exit
  --list-moods           List available moods and exit
```

### Style Conditioning Examples

**Basic Style Selection:**

```bash
# Interactive mode with guided selection
python run_styled_generation.py

# Command line with style selection
python run_advanced_styling.py "mountain landscape" --style cyberpunk --mood dramatic
```

**Advanced Style Control:**

```bash
# High quality with multiple variations
python run_advanced_styling.py "portrait of a wizard" \
  --style "oil painting" --mood "mysterious" \
  --quality high --variations 3 \
  --negative "blurry, distorted, amateur"

# List available options
python run_advanced_styling.py --list-styles
python run_advanced_styling.py --list-moods
```

**Available Styles:**

- digital art, oil painting, watercolor, cyberpunk
- impressionist, concept art, anime, photorealistic
- minimalist, surrealism, pixel art, steampunk, 3d render

**Available Moods:**

- dreamy, dark, peaceful, vibrant, melancholic
- mysterious, whimsical, dramatic, retro

## 🖥️ Phase 1.C: Interactive Web UI

### 5. `compi_phase1c_streamlit_ui.py` - Streamlit Web Interface

**Features:**

- Complete web-based interface for text-to-image generation
- Interactive style and mood selection with custom options
- Advanced settings (steps, guidance, dimensions, negative prompts)
- Real-time image generation and display
- Progress tracking and generation logs
- Automatic saving with comprehensive metadata

**Usage:**

```bash
python run_ui.py
# Or directly: streamlit run src/ui/compi_phase1c_streamlit_ui.py
```

### 6. `compi_phase1c_gradio_ui.py` - Gradio Web Interface

**Features:**

- Alternative web interface with Gradio framework
- Gallery view for multiple image variations
- Collapsible advanced settings
- Real-time generation logs
- Mobile-friendly responsive design

**Usage:**

```bash
python run_gradio_ui.py
# Or directly: python src/ui/compi_phase1c_gradio_ui.py
```

### Web UI Examples

**Streamlit Interface:**

- Navigate to http://localhost:8501 after running
- Full-featured interface with sidebar settings
- Progress bars and status updates
- Expandable sections for details

**Gradio Interface:**

- Navigate to http://localhost:7860 after running
- Gallery-style image display
- Compact, mobile-friendly design
- Real-time generation feedback

## 🎯 Next Steps

Phase 1 establishes the foundation for CompI's text-to-image capabilities. Future phases will add:

- Audio input processing
- Emotion and style conditioning
- Real-time data integration
- Multimodal fusion
- Advanced UI interfaces

## 📚 Resources

- [Stable Diffusion Documentation](https://huggingface.co/docs/diffusers)
- [Prompt Engineering Guide](https://prompthero.com/stable-diffusion-prompt-guide)
- [CompI Development Plan](development.md)

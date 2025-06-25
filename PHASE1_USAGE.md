# CompI Phase 1: Text-to-Image Generation Usage Guide

This guide covers the Phase 1 implementation of CompI's text-to-image generation capabilities using Stable Diffusion.

## 🚀 Quick Start

### Basic Usage
```bash
# Simple generation with interactive prompt
python compi_phase1_text2image.py

# Generate from command line
python compi_phase1_text2image.py "A magical forest, digital art, highly detailed"
```

### Advanced Usage
```bash
# Advanced script with more options
python compi_phase1_advanced.py "cyberpunk city at sunset" --negative "blurry, low quality" --steps 50 --batch 3

# Interactive mode for experimentation
python compi_phase1_advanced.py --interactive
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
python compi_phase1_text2image.py "serene mountain lake, golden hour, photorealistic"

# Digital art style
python compi_phase1_text2image.py "futuristic robot, neon lights, cyberpunk style, digital art"
```

### Advanced Examples
```bash
# High-quality generation with negative prompts
python compi_phase1_advanced.py "beautiful portrait of a woman, oil painting style" \
  --negative "blurry, distorted, low quality, bad anatomy" \
  --steps 50 --guidance 8.0

# Batch generation with fixed seed
python compi_phase1_advanced.py "abstract geometric patterns, colorful" \
  --batch 5 --seed 12345 --steps 40

# Custom dimensions for landscape
python compi_phase1_advanced.py "panoramic view of alien landscape" \
  --width 768 --height 512 --steps 35

# Interactive experimentation
python compi_phase1_advanced.py --interactive
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

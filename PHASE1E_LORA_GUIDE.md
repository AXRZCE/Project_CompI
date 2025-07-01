# CompI Phase 1.E: Personal Style Fine-tuning with LoRA

## 🎯 Overview

Phase 1.E enables you to train **personalized artistic styles** using LoRA (Low-Rank Adaptation) fine-tuning on Stable Diffusion. This allows you to create AI art that reflects your unique artistic vision or mimics specific artistic styles.

**LoRA Benefits:**
- ✅ **Lightweight**: Only trains a small adapter (~10-100MB vs full model ~4GB)
- ✅ **Fast**: Training takes minutes to hours instead of days
- ✅ **Flexible**: Can be combined with different base models
- ✅ **Efficient**: Runs on consumer GPUs (8GB+ VRAM recommended)

## 🛠️ Tools Provided

### 1. **Dataset Preparation** (`compi_phase1e_dataset_prep.py`)
- Organize and validate your style images
- Generate appropriate training captions
- Resize and format images for optimal training
- Create train/validation splits

### 2. **LoRA Training** (`compi_phase1e_lora_training.py`)
- Full LoRA fine-tuning pipeline with PEFT integration
- Configurable training parameters and monitoring
- Automatic checkpoint saving and validation
- Memory-efficient training with gradient checkpointing

### 3. **Style Generation** (`compi_phase1e_style_generation.py`)
- Generate images using your trained LoRA styles
- Interactive and batch generation modes
- Adjustable style strength and parameters
- Integration with existing CompI pipeline

### 4. **Style Management** (`compi_phase1e_style_manager.py`)
- Manage multiple trained LoRA styles
- Cleanup old checkpoints and organize models
- Export style information and analytics
- Switch between different personal styles

## 🚀 Quick Start Guide

### Step 1: Install Dependencies

```bash
# Install LoRA training dependencies
pip install peft datasets bitsandbytes

# Verify installation
python -c "import peft, datasets; print('✅ Dependencies installed')"
```

### Step 2: Prepare Your Style Dataset

```bash
# Organize your style images in a folder
mkdir my_artwork
# Copy 10-50 images of your artistic style to my_artwork/

# Prepare dataset for training
python src/generators/compi_phase1e_dataset_prep.py \
    --input-dir my_artwork \
    --style-name "my_art_style" \
    --trigger-word "myart"
```

**Dataset Requirements:**
- **10-50 images** (more is better, but 20+ is usually sufficient)
- **Consistent style** across all images
- **512x512 pixels** recommended (will be auto-resized)
- **High quality** images without watermarks or text

### Step 3: Train Your LoRA Style

```bash
# Start LoRA training
python run_lora_training.py \
    --dataset-dir datasets/my_art_style \
    --epochs 100 \
    --learning-rate 1e-4

# Or with custom settings
python run_lora_training.py \
    --dataset-dir datasets/my_art_style \
    --epochs 200 \
    --batch-size 2 \
    --lora-rank 8 \
    --lora-alpha 32
```

**Training Tips:**
- **Start with 100 epochs** for initial testing
- **Increase to 200-500 epochs** for stronger style learning
- **Monitor validation loss** to avoid overfitting
- **Use gradient checkpointing** if you run out of memory

### Step 4: Generate with Your Style

```bash
# Generate images with your trained style
python run_style_generation.py \
    --lora-path lora_models/my_art_style/checkpoint-1000 \
    "a cat in myart style" \
    --variations 4

# Interactive mode
python run_style_generation.py \
    --lora-path lora_models/my_art_style/checkpoint-1000 \
    --interactive
```

## 📊 Advanced Usage

### Training Configuration

```bash
# High-quality training (slower but better results)
python run_lora_training.py \
    --dataset-dir datasets/my_style \
    --epochs 300 \
    --learning-rate 5e-5 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --batch-size 1 \
    --gradient-checkpointing

# Fast training (quicker results for testing)
python run_lora_training.py \
    --dataset-dir datasets/my_style \
    --epochs 50 \
    --learning-rate 2e-4 \
    --lora-rank 4 \
    --lora-alpha 16
```

### Style Management

```bash
# List all trained styles
python src/generators/compi_phase1e_style_manager.py --list

# Get detailed info about a style
python src/generators/compi_phase1e_style_manager.py --info my_art_style

# Clean up old checkpoints (keep only 3 most recent)
python src/generators/compi_phase1e_style_manager.py --cleanup my_art_style --keep 3

# Export styles information to CSV
python src/generators/compi_phase1e_style_manager.py --export my_styles_report.csv
```

### Generation Parameters

```bash
# Adjust style strength
python run_style_generation.py \
    --lora-path lora_models/my_style/checkpoint-1000 \
    --lora-scale 0.8 \
    "portrait in myart style"

# High-quality generation
python run_style_generation.py \
    --lora-path lora_models/my_style/checkpoint-1000 \
    --steps 50 \
    --guidance 8.0 \
    --width 768 \
    --height 768 \
    "landscape in myart style"
```

## 🎨 Best Practices

### Dataset Preparation
1. **Consistent Style**: All images should represent the same artistic style
2. **Quality over Quantity**: 20 high-quality images > 100 low-quality ones
3. **Diverse Subjects**: Include various subjects (people, objects, landscapes)
4. **Clean Images**: Remove watermarks, text, and irrelevant elements
5. **Proper Captions**: Use consistent trigger words in captions

### Training Tips
1. **Start Small**: Begin with 50-100 epochs to test
2. **Monitor Progress**: Check validation loss and sample generations
3. **Adjust Learning Rate**: Lower if loss oscillates, higher if learning is slow
4. **Use Checkpoints**: Save frequently to avoid losing progress
5. **Experiment with LoRA Rank**: Higher rank = more capacity but slower training

### Generation Guidelines
1. **Include Trigger Words**: Always use your trigger word in prompts
2. **Adjust Style Strength**: Use `--lora-scale` to control style intensity
3. **Combine with Techniques**: Mix with existing CompI style/mood systems
4. **Iterate and Refine**: Generate multiple variations and select best results

## 🔧 Troubleshooting

### Common Issues

**Out of Memory Error:**
```bash
# Reduce batch size and enable gradient checkpointing
python run_lora_training.py \
    --dataset-dir datasets/my_style \
    --batch-size 1 \
    --gradient-checkpointing \
    --mixed-precision
```

**Style Not Learning:**
- Increase epochs (try 200-500)
- Check dataset consistency
- Increase LoRA rank (try 8 or 16)
- Lower learning rate (try 5e-5)

**Generated Images Don't Match Style:**
- Include trigger word in prompts
- Increase LoRA scale (try 1.2-1.5)
- Train for more epochs
- Check dataset quality

**Training Too Slow:**
- Reduce image resolution to 512x512
- Use mixed precision training
- Enable gradient checkpointing
- Reduce LoRA rank to 4

## 📁 File Structure

```
Project CompI/
├── datasets/                    # Prepared training datasets
│   └── my_art_style/
│       ├── train/              # Training images
│       ├── validation/         # Validation images
│       ├── train_captions.txt  # Training captions
│       └── dataset_info.json   # Dataset metadata
├── lora_models/                # Trained LoRA models
│   └── my_art_style/
│       ├── checkpoint-100/     # Training checkpoints
│       ├── checkpoint-200/
│       └── training_info.json  # Training metadata
├── src/generators/
│   ├── compi_phase1e_dataset_prep.py     # Dataset preparation
│   ├── compi_phase1e_lora_training.py    # LoRA training
│   ├── compi_phase1e_style_generation.py # Style generation
│   └── compi_phase1e_style_manager.py    # Style management
├── run_lora_training.py        # Training launcher
└── run_style_generation.py     # Generation launcher
```

## 🎯 Integration with CompI

Phase 1.E integrates seamlessly with existing CompI tools:

1. **Combine with Phase 1.B**: Use LoRA styles alongside predefined styles
2. **Evaluate with Phase 1.D**: Assess your LoRA-generated images systematically
3. **UI Integration**: Add LoRA styles to Streamlit/Gradio interfaces
4. **Batch Processing**: Generate multiple variations for evaluation

## 🚀 Next Steps

After mastering Phase 1.E:

1. **Experiment with Multiple Styles**: Train different LoRA adapters for various artistic approaches
2. **Style Mixing**: Combine multiple LoRA styles for unique effects
3. **Advanced Techniques**: Explore Textual Inversion, DreamBooth, or ControlNet integration
4. **Community Sharing**: Share your trained styles with the CompI community
5. **Phase 2 Preparation**: Use personal styles as foundation for multimodal integration

---

**Happy Style Training! 🎨✨**

Phase 1.E opens up endless possibilities for personalized AI art generation. With LoRA fine-tuning, you can teach the AI to understand and replicate your unique artistic vision, creating truly personalized creative content.

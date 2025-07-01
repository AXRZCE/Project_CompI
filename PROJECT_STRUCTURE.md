# CompI Project Structure

This document outlines the organized structure of the CompI (Compositional Intelligence) project.

## 📁 Directory Structure

```
Project CompI/
├── 📁 src/                          # Source code (organized modules)
│   ├── 📁 generators/               # Image generation modules
│   │   ├── __init__.py             # Module initialization
│   │   ├── compi_phase1_text2image.py          # Basic text-to-image
│   │   ├── compi_phase1_advanced.py            # Advanced generation
│   │   ├── compi_phase1b_styled_generation.py  # Style conditioning
│   │   ├── compi_phase1b_advanced_styling.py   # Advanced styling
│   │   ├── compi_phase1d_evaluate_quality.py   # Quality evaluation (Streamlit)
│   │   ├── compi_phase1d_cli_evaluation.py     # Quality evaluation (CLI)
│   │   ├── compi_phase1e_dataset_prep.py       # LoRA dataset preparation
│   │   ├── compi_phase1e_lora_training.py      # LoRA fine-tuning
│   │   ├── compi_phase1e_style_generation.py   # Personal style generation
│   │   └── compi_phase1e_style_manager.py      # LoRA style management
│   ├── 📁 models/                   # Model implementations (future)
│   ├── 📁 utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── logging_utils.py
│   │   └── file_utils.py
│   ├── 📁 data/                     # Data processing modules (future)
│   ├── 📁 ui/                       # User interface components (future)
│   ├── config.py                    # Project configuration
│   ├── setup_env.py                 # Environment setup script
│   └── test_setup.py                # Environment testing
├── 📁 notebooks/                    # Jupyter notebooks
│   └── 01_getting_started.ipynb    # Tutorial notebook
├── 📁 data/                         # Dataset storage
├── 📁 outputs/                      # Generated content
│   ├── images/                      # Generated images
│   └── metadata/                    # Generation metadata
├── 📁 tests/                        # Unit tests (future)
├── 🐍 run_basic_generation.py       # Convenience: Basic generation
├── 🐍 run_advanced_generation.py    # Convenience: Advanced generation
├── 🐍 run_styled_generation.py      # Convenience: Style conditioning
├── 🐍 run_advanced_styling.py       # Convenience: Advanced styling
├── 🐍 run_evaluation.py             # Convenience: Quality evaluation
├── 🐍 run_lora_training.py          # Convenience: LoRA training
├── 🐍 run_style_generation.py       # Convenience: Personal style generation
├── 📄 requirements.txt              # Python dependencies
├── 📄 development.md                # Development roadmap
├── 📄 PHASE1_USAGE.md              # Phase 1 usage guide
├── 📄 PROJECT_STRUCTURE.md         # This file
├── 📄 .gitignore                   # Git ignore rules
└── 📄 README.md                    # Project overview
```

## 🚀 Usage Patterns

### Convenience Scripts (Recommended)

Run from project root for easy access:

```bash
# Basic text-to-image generation
python run_basic_generation.py "prompt here"

# Advanced generation with options
python run_advanced_generation.py "prompt" --negative "unwanted" --steps 50

# Interactive style selection
python run_styled_generation.py

# Advanced style conditioning
python run_advanced_styling.py "prompt" --style "oil painting" --mood "dramatic"

# Quality evaluation interface
python run_evaluation.py

# LoRA personal style training
python run_lora_training.py --dataset-dir datasets/my_style

# Generate with personal style
python run_style_generation.py --lora-path lora_models/my_style/checkpoint-1000 "prompt"

# LoRA personal style training
python run_lora_training.py --dataset-dir datasets/my_style

# Generate with personal style
python run_style_generation.py --lora-path lora_models/my_style/checkpoint-1000 "prompt"
```

### Direct Module Access

Run generators directly from their organized location:

```bash
# Direct access to generators
python src/generators/compi_phase1_text2image.py "prompt"
python src/generators/compi_phase1b_advanced_styling.py --list-styles

# Environment setup and testing
python src/setup_env.py
python src/test_setup.py
```

## 🎯 Benefits of This Organization

### 1. **Clean Separation of Concerns**

- **`src/generators/`** - All image generation logic
- **`src/utils/`** - Reusable utility functions
- **`src/`** - Core project modules and configuration
- **Root level** - Convenience scripts and documentation

### 2. **Professional Python Structure**

- Proper module organization with `__init__.py` files
- Clear import paths and dependencies
- Scalable architecture for future expansion

### 3. **Easy Access**

- Convenience scripts provide simple access from project root
- Direct module access for advanced users
- Maintains backward compatibility

### 4. **Future-Ready**

- Organized structure ready for Phase 2+ implementations
- Clear places for audio processing, UI components, etc.
- Modular design supports easy testing and maintenance

## 🔧 Development Guidelines

### Adding New Generators

1. Create new generator in `src/generators/`
2. Add imports to `src/generators/__init__.py`
3. Create convenience script in project root if needed
4. Update documentation

### Adding New Utilities

1. Add utility functions to appropriate module in `src/utils/`
2. Update `src/utils/__init__.py` imports
3. Import in generators as needed

### Testing

1. Add tests to `tests/` directory
2. Use `src/test_setup.py` for environment verification
3. Test both convenience scripts and direct module access

## 📚 Documentation

- **README.md** - Project overview and quick start
- **development.md** - Comprehensive development roadmap
- **PHASE1_USAGE.md** - Detailed Phase 1 usage guide
- **PROJECT_STRUCTURE.md** - This organizational guide

This structure provides a solid foundation for the CompI project's continued development through all planned phases.

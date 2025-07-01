# CompI Phase 2.A: Audio-to-Image Generation Guide

Welcome to **CompI Phase 2.A**, the next evolution in multimodal AI art generation! This phase introduces the ability to generate images influenced by audio input, combining the power of text prompts with the emotional and rhythmic qualities of sound.

## 🎵 What's New in Phase 2.A

### Core Features
- **Audio Analysis**: Extract tempo, energy, spectral features, and harmonic content from audio files
- **Audio Captioning**: Convert speech, music, and ambient sounds to descriptive text using OpenAI Whisper
- **Multimodal Fusion**: Intelligently combine text prompts with audio-derived features
- **Rich Metadata**: Comprehensive logging of audio features and generation context
- **Multiple Interfaces**: Streamlit UI, CLI, and programmatic API

### Supported Audio Formats
- MP3, WAV, FLAC, M4A, OGG
- Recommended: Under 60 seconds for optimal processing speed
- Automatic resampling to 16kHz for analysis

## 🚀 Quick Start

### 1. Install Dependencies

First, ensure you have the Phase 2.A dependencies:

```bash
pip install openai-whisper
```

All other dependencies should already be installed from Phase 1.

### 2. Streamlit UI (Recommended for Beginners)

Launch the interactive web interface:

```bash
streamlit run src/ui/compi_phase2a_streamlit_ui.py
```

Features:
- 🎵 Audio upload and playback
- 📊 Real-time audio analysis visualization
- 🎨 Interactive generation controls
- 📝 Enhanced prompt preview
- 🖼️ Instant results display

### 3. Command Line Interface

For power users and automation:

```bash
# Basic usage
python run_phase2a_audio_to_image.py --prompt "mystical forest" --audio "music.mp3"

# With style and mood
python run_phase2a_audio_to_image.py \
    --prompt "cyberpunk city" \
    --style "digital art" \
    --mood "neon, futuristic" \
    --audio "electronic.wav"

# Multiple variations
python run_phase2a_audio_to_image.py \
    --prompt "abstract art" \
    --audio "ambient.flac" \
    --num-images 3

# Interactive mode
python run_phase2a_audio_to_image.py --interactive
```

### 4. Programmatic Usage

```python
from src.generators.compi_phase2a_audio_to_image import CompIPhase2AAudioToImage

# Initialize generator
generator = CompIPhase2AAudioToImage()

# Generate image with audio conditioning
results = generator.generate_image(
    text_prompt="A serene mountain landscape",
    style="impressionist",
    mood="peaceful, contemplative",
    audio_path="nature_sounds.wav",
    num_images=2
)

# Access results
for result in results:
    print(f"Generated: {result['filename']}")
    result['image'].show()  # Display image
```

## 🎨 How Audio Influences Art

### Audio Feature Extraction

CompI Phase 2.A analyzes multiple aspects of your audio:

1. **Tempo**: Beats per minute → influences rhythm and energy descriptors
2. **Energy (RMS)**: Overall loudness → affects intensity and power descriptors
3. **Zero Crossing Rate**: Rhythmic content → adds percussive/smooth qualities
4. **Spectral Centroid**: Brightness → influences warm/bright color palettes
5. **MFCC**: Timbre characteristics → affects texture and style
6. **Chroma**: Harmonic content → influences mood and atmosphere

### Intelligent Prompt Fusion

The system automatically enhances your text prompt based on audio analysis:

**Original Prompt**: "A mystical forest"
**Audio**: Slow, ambient music with low energy
**Enhanced Prompt**: "A mystical forest, slow and contemplative, gentle and subtle, warm and deep"

### Audio Captioning

Using OpenAI Whisper, the system can describe what it "hears":
- **Speech**: Transcribes spoken words and incorporates meaning
- **Music**: Identifies instruments, genres, and emotional qualities
- **Ambient**: Describes environmental sounds and atmospheres

## 📊 Understanding Audio Analysis

### Tempo Classifications
- **Very Slow** (< 60 BPM): Meditative, ethereal qualities
- **Slow** (60-90 BPM): Contemplative, peaceful atmospheres
- **Moderate** (90-120 BPM): Balanced, natural rhythms
- **Fast** (120-140 BPM): Energetic, dynamic compositions
- **Very Fast** (> 140 BPM): Intense, high-energy visuals

### Energy Levels
- **Low Energy** (< 0.02): Subtle, gentle, minimalist styles
- **Medium Energy** (0.02-0.05): Balanced, harmonious compositions
- **High Energy** (> 0.05): Vibrant, powerful, dramatic visuals

### Spectral Characteristics
- **Bright** (High Spectral Centroid): Light colors, sharp details
- **Dark** (Low Spectral Centroid): Deep colors, soft textures
- **Percussive** (High ZCR): Rhythmic patterns, geometric shapes
- **Smooth** (Low ZCR): Flowing forms, organic shapes

## 🎯 Best Practices

### Audio Selection
1. **Quality Matters**: Use clear, well-recorded audio for best results
2. **Length**: 10-60 seconds is optimal for processing speed
3. **Variety**: Experiment with different genres and sound types
4. **Context**: Choose audio that complements your text prompt

### Prompt Writing
1. **Be Descriptive**: Rich text prompts work better with audio conditioning
2. **Leave Room**: Let audio features add nuance to your base concept
3. **Experiment**: Try the same prompt with different audio files
4. **Balance**: Don't over-specify if you want audio to have strong influence

### Generation Settings
1. **Steps**: 30-50 steps for high quality (20 for quick tests)
2. **Guidance**: 7.5 is balanced (lower for more audio influence)
3. **Variations**: Generate multiple images to see different interpretations
4. **Seeds**: Save seeds of favorite results for consistency

## 🔧 Advanced Features

### Batch Processing

Process multiple audio files with the same prompt:

```bash
python run_phase2a_audio_to_image.py \
    --prompt "abstract expressionism" \
    --audio-dir "./music_collection/" \
    --batch
```

### Custom Audio Analysis

```python
from src.utils.audio_utils import AudioProcessor, MultimodalPromptFusion

# Analyze audio separately
processor = AudioProcessor()
features = processor.analyze_audio_file("my_audio.wav")

# Create custom prompt fusion
fusion = MultimodalPromptFusion()
enhanced_prompt = fusion.fuse_prompt_with_audio(
    "base prompt", "style", "mood", features, "audio caption"
)
```

### Metadata and Tracking

Every generated image includes comprehensive metadata:
- Original and enhanced prompts
- Complete audio analysis results
- Generation parameters
- Timestamps and seeds
- Audio tags and classifications

## 🎪 Example Use Cases

### 1. Music Visualization
Transform your favorite songs into visual art:
- **Classical**: Orchestral pieces → elegant, flowing compositions
- **Electronic**: Synthesized music → geometric, neon aesthetics
- **Jazz**: Improvisational music → abstract, dynamic forms
- **Ambient**: Atmospheric sounds → ethereal, dreamlike scenes

### 2. Voice-to-Art
Convert spoken content into visuals:
- **Poetry Reading**: Emotional recitation → expressive, literary art
- **Storytelling**: Narrative audio → scene illustrations
- **Meditation**: Guided meditation → peaceful, spiritual imagery
- **Lectures**: Educational content → informative, structured visuals

### 3. Environmental Soundscapes
Capture the essence of places and moments:
- **Nature Sounds**: Forest, ocean, rain → organic, natural scenes
- **Urban Audio**: City sounds, traffic → industrial, modern aesthetics
- **Historical**: Period-appropriate audio → era-specific artwork
- **Sci-Fi**: Futuristic sounds → otherworldly, technological visuals

### 4. Therapeutic Applications
Use audio-visual generation for wellness:
- **Relaxation**: Calming audio → soothing, peaceful imagery
- **Motivation**: Energetic music → inspiring, powerful visuals
- **Focus**: Concentration aids → clean, organized compositions
- **Creativity**: Experimental sounds → abstract, innovative art

## 🐛 Troubleshooting

### Common Issues

**Audio Not Loading**
- Check file format (MP3, WAV, FLAC, M4A, OGG supported)
- Ensure file isn't corrupted
- Try converting to WAV format

**Whisper Model Loading Fails**
- Install with: `pip install openai-whisper`
- Check available disk space (models are 100MB-1GB)
- Try smaller model size: `--whisper-model tiny`

**Generation Too Slow**
- Use `--no-caption` to skip audio captioning
- Reduce `--steps` for faster generation
- Use smaller Whisper model
- Process shorter audio clips

**Out of Memory**
- Use CPU mode: `--device cpu`
- Reduce image size: `--size 256x256`
- Close other applications
- Process one image at a time

### Performance Tips

1. **GPU Acceleration**: CUDA significantly speeds up generation
2. **Model Caching**: First run downloads models (1-2GB total)
3. **Audio Preprocessing**: Shorter clips process faster
4. **Batch Processing**: More efficient for multiple files
5. **Memory Management**: Close UI between large batches

## 🔮 What's Next?

Phase 2.A is just the beginning of CompI's multimodal journey. Coming soon:

- **Phase 2.B**: Real-time audio processing and live generation
- **Phase 2.C**: Video-to-image conditioning
- **Phase 2.D**: Multi-sensor input fusion
- **Phase 3.A**: 3D model generation from multimodal input

## 📚 Additional Resources

- [CompI Project Structure](PROJECT_STRUCTURE.md)
- [Phase 1 Usage Guide](PHASE1_USAGE.md)
- [Audio Processing Documentation](src/utils/audio_utils.py)
- [Example Audio Files](examples/audio_samples/)

---

**Happy Creating! 🎨🎵**

*CompI Phase 2.A brings together the worlds of sound and vision, creating art that truly resonates with your audio experiences.*

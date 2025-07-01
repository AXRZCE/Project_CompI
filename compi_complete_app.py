"""
CompI Complete Application - All Phases in One Streamlit App

This is the unified interface for the entire CompI project, including:
- Phase 1: Text-to-Image Generation
- Phase 2.A: Audio-to-Image Generation
- Phase 2.B: Data/Logic-to-Image Generation
- Phase 2.C: Emotional/Contextual Input to Image Generation
- Phase 2.D: Real-Time Data Feeds to Image Generation
- Project Overview and Documentation
- Image Gallery and Management
"""

import os
import sys
import streamlit as st
import torch
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(__file__))

# Page configuration
st.set_page_config(
    page_title="CompI: Complete Multimodal AI Art Platform",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .phase-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2e8b57;
        margin: 1rem 0;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("# 🎨 CompI Navigation")

# Device information
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"🖥️ Device: {device.upper()}")
if device == "cuda":
    try:
        gpu_name = torch.cuda.get_device_name(0)
        st.sidebar.success(f"🚀 GPU: {gpu_name}")
    except:
        pass

# Navigation menu
page = st.sidebar.selectbox(
    "Choose a page:",
    [
        "🏠 Project Overview",
        "🎨 Phase 1: Text-to-Image",
        "🎵 Phase 2.A: Audio-to-Image",
        "📊 Phase 2.B: Data-to-Image",
        "🌀 Phase 2.C: Emotion-to-Image",
        "🌎 Phase 2.D: Real-Time Data-to-Image",
        "🖼️ Image Gallery",
        "📈 Analytics & Evaluation",
        "📚 Documentation",
        "⚙️ Settings & Tools"
    ]
)

# Main header
st.markdown('<h1 class="main-header">🎨 CompI: Complete Multimodal AI Art Platform</h1>', unsafe_allow_html=True)

# Page routing
if page == "🏠 Project Overview":
    st.markdown("## Welcome to CompI - The Future of AI Art Generation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **CompI** is a cutting-edge multimodal AI art generation platform that combines text, audio,
        data, emotions, real-time feeds, and advanced styling to create unique visual art. Our platform represents the evolution of
        creative AI, moving beyond simple text-to-image generation to truly multimodal experiences.
        """)
        
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 🚀 **Current Capabilities**
        
        **Phase 1: Advanced Text-to-Image Generation**
        - Multiple art styles (digital art, oil painting, watercolor, etc.)
        - Mood and atmosphere conditioning
        - Quality evaluation and analysis
        - LoRA fine-tuning for personal styles
        - Batch processing capabilities
        
        **Phase 2.A: Revolutionary Audio-to-Image Generation**
        - Audio analysis (tempo, energy, spectral features)
        - Speech-to-text captioning with OpenAI Whisper
        - Intelligent multimodal prompt fusion
        - Real-time audio visualization
        - Support for multiple audio formats

        **Phase 2.B: Data/Logic-to-Image Generation**
        - CSV data analysis and pattern recognition
        - Mathematical formula evaluation and visualization
        - Poetic interpretation of data patterns
        - Statistical analysis and trend detection
        - Safe mathematical expression processing
        - Data-driven artistic conditioning

        **Phase 2.C: Emotional/Contextual Input to Image Generation**
        - Emotion detection and sentiment analysis
        - Preset emotions, custom emotions, and emoji support
        - Contextual mood processing and interpretation
        - Color palette generation based on emotions
        - Intelligent emotional prompt enhancement
        - Batch processing for multiple emotional states

        **Phase 2.D: Real-Time Data Feeds to Image Generation**
        - Live weather data integration from multiple APIs
        - Breaking news headlines and RSS feed processing
        - Financial market data and cryptocurrency prices
        - Real-time context analysis and summarization
        - Temporal series generation for data evolution
        - Intelligent fusion of real-time data with creative prompts
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 📈 **Project Stats**

        **Phases Completed:** 5/5

        **Supported Formats:**
        - Images: PNG with metadata
        - Audio: MP3, WAV, FLAC, M4A, OGG
        - Data: CSV files, Mathematical formulas
        - Emotions: Preset emotions, Custom text, Emojis
        - Real-Time: Weather, News, Financial data
        - Styles: 13+ predefined + custom
        
        **Performance:**
        - GPU Acceleration: ✅
        - Real-time Processing: ✅
        - Batch Generation: ✅
        
        **Documentation:**
        - User Guides: ✅
        - API Documentation: ✅
        - Examples: ✅
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick start section
    st.markdown("### 🎯 Quick Start")
    
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        if st.button("🎨 Start Text-to-Image", use_container_width=True):
            st.session_state.page = "🎨 Phase 1: Text-to-Image"
            st.rerun()
    
    with quick_col2:
        if st.button("🎵 Try Audio-to-Image", use_container_width=True):
            st.session_state.page = "🎵 Phase 2.A: Audio-to-Image"
            st.rerun()
    
    with quick_col3:
        if st.button("🖼️ View Gallery", use_container_width=True):
            st.session_state.page = "🖼️ Image Gallery"
            st.rerun()
    
    # Recent activity
    st.markdown("### 📊 Recent Activity")
    
    try:
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            recent_files = sorted(
                [f for f in outputs_dir.glob("*.png")],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )[:6]
            
            if recent_files:
                cols = st.columns(3)
                for i, file_path in enumerate(recent_files):
                    with cols[i % 3]:
                        try:
                            from PIL import Image
                            img = Image.open(file_path)
                            st.image(img, caption=file_path.stem[:30] + "...", use_container_width=True)
                        except:
                            st.write(f"📄 {file_path.name}")
            else:
                st.info("No images generated yet. Start creating!")
        else:
            st.info("No outputs directory found. Generate your first image!")
    except Exception as e:
        st.warning(f"Could not load recent activity: {e}")

elif page == "🎨 Phase 1: Text-to-Image":
    st.markdown('<h2 class="phase-header">🎨 Phase 1: Advanced Text-to-Image Generation</h2>', unsafe_allow_html=True)
    
    # Import Phase 1 components
    try:
        # Text-to-image interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📝 Generation Settings")
            
            text_prompt = st.text_area(
                "Text Prompt",
                value="A majestic dragon soaring through a crystal cave",
                height=100,
                help="Describe what you want to see in detail"
            )
            
            col_style, col_mood = st.columns(2)
            with col_style:
                style_options = [
                    "digital art", "oil painting", "watercolor", "cyberpunk", 
                    "impressionist", "concept art", "anime", "photorealistic",
                    "minimalist", "surrealism", "pixel art", "steampunk", "3d render"
                ]
                style = st.selectbox("Art Style", style_options, index=0)
            
            with col_mood:
                mood_options = [
                    "epic and dramatic", "peaceful and serene", "dark and moody",
                    "bright and cheerful", "mysterious", "whimsical", "futuristic",
                    "vintage", "ethereal", "powerful", "gentle"
                ]
                mood = st.selectbox("Mood/Atmosphere", mood_options, index=0)
        
        with col2:
            st.markdown("### ⚙️ Advanced Settings")
            
            num_images = st.slider("Number of Images", 1, 4, 1)
            num_steps = st.slider("Quality Steps", 10, 50, 30, step=5)
            guidance_scale = st.slider("Prompt Adherence", 1.0, 20.0, 7.5, step=0.5)
            
            image_size = st.selectbox(
                "Image Size",
                ["512x512", "768x768", "512x768", "768x512"],
                index=0
            )
            
            seed = st.number_input(
                "Seed (optional)",
                min_value=0,
                max_value=2**32-1,
                value=0,
                help="Use 0 for random seed"
            )
        
        # Generation button
        if st.button("🎨 Generate Images", type="primary", use_container_width=True):
            if not text_prompt.strip():
                st.error("Please enter a text prompt!")
            else:
                with st.spinner(f"Generating {num_images} image(s)... This may take a few minutes."):
                    try:
                        # Parse image size
                        width, height = map(int, image_size.split('x'))
                        
                        # Import and use the generator
                        from src.generators.compi_phase2a_audio_to_image import CompIPhase2AAudioToImage
                        
                        generator = CompIPhase2AAudioToImage(output_dir="outputs")
                        
                        results = generator.generate_image(
                            text_prompt=text_prompt,
                            style=style,
                            mood=mood,
                            audio_path=None,  # No audio for Phase 1
                            num_images=num_images,
                            height=height,
                            width=width,
                            num_inference_steps=num_steps,
                            guidance_scale=guidance_scale,
                            seed=seed if seed > 0 else None
                        )
                        
                        if results:
                            st.success(f"✅ Generated {len(results)} image(s) successfully!")
                            
                            # Display results
                            if len(results) == 1:
                                result = results[0]
                                st.image(result["image"], caption=result["filename"], use_container_width=True)
                                st.info(f"💾 Saved as: `{result['filename']}.png`")
                            else:
                                cols = st.columns(min(len(results), 3))
                                for i, result in enumerate(results):
                                    with cols[i % len(cols)]:
                                        st.image(result["image"], caption=f"Variation {i+1}", use_container_width=True)
                                        st.caption(result["filename"])
                            
                            # Show generation details
                            with st.expander("📋 Generation Details", expanded=False):
                                for i, result in enumerate(results):
                                    st.write(f"**Image {i+1}:**")
                                    st.json(result["metadata"]["generation_params"])
                        
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
                        st.exception(e)
    
    except Exception as e:
        st.error(f"Error loading Phase 1 interface: {e}")

elif page == "🎵 Phase 2.A: Audio-to-Image":
    st.markdown('<h2 class="phase-header">🎵 Phase 2.A: Revolutionary Audio-to-Image Generation</h2>', unsafe_allow_html=True)

    # Audio-to-image interface
    import tempfile

    col1, col2 = st.columns([2, 1])

    with col1:
        text_prompt = st.text_input(
            "Text Prompt",
            value="A mystical landscape with flowing energy",
            help="Describe what you want to see in the image"
        )

        style = st.text_input(
            "Art Style",
            value="digital art",
            help="e.g., oil painting, watercolor, cyberpunk, impressionist"
        )

        mood = st.text_input(
            "Mood/Atmosphere",
            value="ethereal, dreamlike",
            help="e.g., dark and moody, bright and cheerful, mysterious"
        )

    with col2:
        num_images = st.number_input(
            "Number of Images",
            min_value=1, max_value=4, value=1, step=1
        )

        num_steps = st.slider(
            "Quality Steps",
            min_value=10, max_value=50, value=20, step=5,
            help="More steps = higher quality but slower generation"
        )

    # Audio section
    st.header("🎵 Audio Input")

    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=["mp3", "wav", "flac", "m4a", "ogg"],
        help="Upload music, voice, or ambient sound"
    )

    audio_features = None
    audio_caption = ""

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_audio_path = tmp_file.name

        # Analyze audio
        with st.spinner("Analyzing audio... 🎵"):
            try:
                # Import here to avoid module-level issues
                from src.utils.audio_utils import AudioProcessor, MultimodalPromptFusion

                processor = AudioProcessor()
                features = processor.analyze_audio_file(temp_audio_path)

                st.success("✅ Audio analysis complete!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Duration", f"{features.duration:.1f}s")
                    st.metric("Tempo", f"{features.tempo:.1f} BPM")
                with col2:
                    st.metric("Energy", f"{features.energy:.4f}")
                    st.metric("Brightness", f"{features.spectral_centroid:.0f} Hz")
                with col3:
                    st.metric("Rhythm", f"{features.zero_crossing_rate:.3f}")
                    st.metric("Sample Rate", f"{features.sample_rate} Hz")

                # Generate audio tags
                fusion = MultimodalPromptFusion()
                audio_tags = fusion.generate_audio_tags(features)
                st.info(f"🏷️ **Audio Tags:** {', '.join(audio_tags)}")

                audio_features = features

            except Exception as e:
                st.error(f"Error analyzing audio: {e}")
                audio_features = None

        # Clean up temp file
        try:
            os.remove(temp_audio_path)
        except:
            pass

    # Prompt preview
    st.header("📝 Enhanced Prompt Preview")

    if audio_features is not None:
        try:
            from src.utils.audio_utils import MultimodalPromptFusion
            fusion = MultimodalPromptFusion()
            enhanced_prompt = fusion.fuse_prompt_with_audio(
                text_prompt, style, mood, audio_features, audio_caption
            )
        except:
            enhanced_prompt = f"{text_prompt}, {style}, {mood}"
    else:
        enhanced_prompt = f"{text_prompt}, {style}, {mood}" if style or mood else text_prompt

    st.markdown(f"**Enhanced Prompt:** `{enhanced_prompt}`")

    # Generation section
    st.header("🎨 Generate Images")

    if st.button("🚀 Generate Art", type="primary", use_container_width=True):
        if not text_prompt.strip():
            st.error("Please enter a text prompt!")
        else:
            with st.spinner(f"Generating {num_images} image(s)... This may take a few minutes."):
                try:
                    # Import generator here to avoid module-level issues
                    from src.generators.compi_phase2a_audio_to_image import CompIPhase2AAudioToImage

                    # Re-save audio file if needed
                    audio_path_for_generation = None
                    if uploaded_file is not None:
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            audio_path_for_generation = tmp_file.name

                    # Initialize generator
                    generator = CompIPhase2AAudioToImage(
                        output_dir="outputs",
                        whisper_model="tiny"
                    )

                    # Generate images
                    results = generator.generate_image(
                        text_prompt=text_prompt,
                        style=style,
                        mood=mood,
                        audio_path=audio_path_for_generation,
                        num_images=num_images,
                        num_inference_steps=num_steps,
                        height=512,
                        width=512
                    )

                    # Clean up temp file
                    if audio_path_for_generation:
                        try:
                            os.remove(audio_path_for_generation)
                        except:
                            pass

                    if results:
                        st.success(f"✅ Generated {len(results)} image(s) successfully!")

                        # Display results
                        if len(results) == 1:
                            result = results[0]
                            st.image(result["image"], caption=result["filename"], use_container_width=True)
                            st.info(f"💾 Saved as: `{result['filename']}.png`")
                        else:
                            cols = st.columns(min(len(results), 3))
                            for i, result in enumerate(results):
                                with cols[i % len(cols)]:
                                    st.image(result["image"], caption=f"Variation {i+1}", use_container_width=True)
                                    st.caption(result["filename"])

                        # Show generation details
                        with st.expander("📋 Generation Details", expanded=False):
                            for i, result in enumerate(results):
                                st.write(f"**Image {i+1}:**")
                                st.json({
                                    "enhanced_prompt": result["metadata"]["enhanced_prompt"],
                                    "generation_params": result["metadata"]["generation_params"]
                                })
                                if "audio_features" in result["metadata"]:
                                    st.write(f"Audio Tags: {', '.join(result['metadata']['audio_tags'])}")

                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    st.exception(e)

elif page == "📊 Phase 2.B: Data-to-Image":
    st.markdown('<h2 class="phase-header">📊 Phase 2.B: Data/Logic-to-Image Generation</h2>', unsafe_allow_html=True)

    st.markdown("""
    **Transform structured data and mathematical formulas into stunning AI art!**
    Upload CSV files or enter mathematical expressions to create unique visualizations
    that blend data insights with artistic creativity.
    """)

    # Data-to-image interface
    import tempfile
    import pandas as pd
    import numpy as np

    col1, col2 = st.columns([2, 1])

    with col1:
        text_prompt = st.text_input(
            "Text Prompt",
            value="A visualization of nature's mathematical patterns",
            help="Describe what you want to see in the image"
        )

        style = st.text_input(
            "Art Style",
            value="digital art",
            help="e.g., abstract, geometric, organic, minimalist"
        )

        mood = st.text_input(
            "Mood/Atmosphere",
            value="harmonious, flowing",
            help="e.g., dynamic, peaceful, mysterious, energetic"
        )

    with col2:
        num_images = st.number_input(
            "Number of Images",
            min_value=1, max_value=4, value=1, step=1
        )

        num_steps = st.slider(
            "Quality Steps",
            min_value=10, max_value=50, value=30, step=5,
            help="More steps = higher quality but slower generation"
        )

        guidance_scale = st.slider(
            "Prompt Adherence",
            min_value=1.0, max_value=20.0, value=7.5, step=0.5,
            help="Higher values follow the prompt more closely"
        )

    # Data input section
    st.header("📈 Data/Logic Input")

    input_type = st.radio(
        "Choose Input Type:",
        ["CSV Data Upload", "Mathematical Formula"],
        help="Select whether to upload data or enter a mathematical formula"
    )

    data_features = None
    poetic_description = ""
    data_visualization = None
    csv_path = None
    formula = None

    if input_type == "CSV Data Upload":
        st.subheader("📊 Upload CSV Data")

        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            help="Upload a CSV file with numeric data"
        )

        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.read())
                csv_path = tmp_file.name

            # Display data preview
            df = pd.read_csv(csv_path)
            st.write("**Data Preview:**")
            st.dataframe(df.head(10))

            # Analyze data
            with st.spinner("Analyzing data patterns... 📊"):
                try:
                    from src.generators.compi_phase2b_data_to_image import CompIPhase2BDataToImage

                    generator = CompIPhase2BDataToImage()
                    df, data_features, poetic_description, data_visualization = generator.analyze_csv_data(csv_path)

                    st.success("✅ Data analysis complete!")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", data_features.shape[0])
                        st.metric("Columns", data_features.shape[1])
                    with col2:
                        st.metric("Numeric Columns", len(data_features.numeric_columns))
                        st.metric("Complexity Score", f"{data_features.complexity_score:.3f}")
                    with col3:
                        st.metric("Variability Score", f"{data_features.variability_score:.3f}")
                        st.metric("Pattern Strength", f"{data_features.pattern_strength:.3f}")

                    # Show data visualization
                    if data_visualization:
                        st.image(data_visualization, caption="Data Pattern Visualization", use_column_width=True)

                    # Show poetic description
                    if poetic_description:
                        st.info(f"🎭 **Data Poetry:** {poetic_description}")

                except Exception as e:
                    st.error(f"Error analyzing data: {e}")
                    data_features = None

    else:  # Mathematical Formula
        st.subheader("🧮 Mathematical Formula")

        # Formula examples
        with st.expander("📚 Formula Examples", expanded=False):
            st.code("""
Examples:
• Sine wave: np.sin(np.linspace(0, 4*np.pi, 100))
• Spiral: np.sin(np.linspace(0, 10, 100)) * np.exp(-np.linspace(0, 2, 100))
• Polynomial: np.linspace(-5, 5, 100)**3 - 3*np.linspace(-5, 5, 100)
• Complex: np.sin(x) + 0.5*np.cos(3*x) where x = np.linspace(0, 6*np.pi, 200)
            """)

        formula = st.text_area(
            "Enter Mathematical Formula",
            value="np.sin(np.linspace(0, 4*np.pi, 100)) * np.exp(-np.linspace(0, 1, 100))",
            height=100,
            help="Use Python/NumPy syntax"
        )

        if formula.strip():
            with st.spinner("Evaluating formula... 🧮"):
                try:
                    from src.generators.compi_phase2b_data_to_image import CompIPhase2BDataToImage

                    generator = CompIPhase2BDataToImage()
                    result_array, metadata, poetic_description, data_visualization = generator.evaluate_mathematical_formula(formula)

                    # Debug information
                    st.write(f"**Debug Info:**")
                    st.write(f"- Result array shape: {result_array.shape}")
                    st.write(f"- Metadata keys: {list(metadata.keys())}")
                    st.write(f"- Poetic description length: {len(poetic_description) if poetic_description else 0}")
                    st.write(f"- Visualization type: {type(data_visualization)}")
                    if data_visualization:
                        st.write(f"- Visualization size: {data_visualization.size}")

                    st.success("✅ Formula evaluation complete!")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Data Points", metadata['length'])
                        st.metric("Min Value", f"{metadata['min']:.4f}")
                    with col2:
                        st.metric("Max Value", f"{metadata['max']:.4f}")
                        st.metric("Mean", f"{metadata['mean']:.4f}")
                    with col3:
                        st.metric("Range", f"{metadata['range']:.4f}")
                        st.metric("Has Pattern", "Yes" if metadata['has_pattern'] else "No")

                    # Show formula visualization
                    if data_visualization:
                        st.image(data_visualization, caption="Mathematical Pattern Visualization", use_column_width=True)

                    # Show poetic description
                    if poetic_description:
                        st.info(f"🎭 **Mathematical Poetry:** {poetic_description}")

                except Exception as e:
                    st.error(f"Error evaluating formula: {e}")
                    formula = None

    # Prompt preview
    st.header("📝 Enhanced Prompt Preview")

    enhanced_prompt = text_prompt
    if style:
        enhanced_prompt += f", {style}"
    if mood:
        enhanced_prompt += f", {mood}"
    if poetic_description:
        enhanced_prompt += f", {poetic_description}"

    st.markdown(f"**Enhanced Prompt:** `{enhanced_prompt}`")

    # Generation section
    st.header("🎨 Generate Images")

    if st.button("🚀 Generate Data Art", type="primary", use_container_width=True):
        if not text_prompt.strip():
            st.error("Please enter a text prompt!")
        elif not csv_path and not formula:
            st.error("Please upload CSV data or enter a mathematical formula!")
        else:
            with st.spinner(f"Generating {num_images} image(s)... This may take a few minutes."):
                try:
                    from src.generators.compi_phase2b_data_to_image import CompIPhase2BDataToImage

                    # Initialize generator
                    generator = CompIPhase2BDataToImage(output_dir="outputs")

                    # Generate images
                    results = generator.generate_image(
                        text_prompt=text_prompt,
                        style=style,
                        mood=mood,
                        csv_path=csv_path,
                        formula=formula,
                        num_images=num_images,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        save_data_visualization=True
                    )

                    if results:
                        st.success(f"✅ Generated {len(results)} image(s) successfully!")

                        # Display results
                        if len(results) == 1:
                            result = results[0]
                            col1, col2 = st.columns([2, 1])

                            with col1:
                                st.image(result["image"], caption=result["filename"], use_column_width=True)
                                st.info(f"💾 Saved as: `{result['filename']}.png`")

                            with col2:
                                if result["data_visualization"]:
                                    st.image(result["data_visualization"], caption="Data Pattern", use_column_width=True)

                                if result["poetic_description"]:
                                    st.markdown("**🎭 Data Poetry:**")
                                    st.write(result["poetic_description"])
                        else:
                            # Multiple images
                            cols = st.columns(min(len(results), 3))
                            for i, result in enumerate(results):
                                with cols[i % len(cols)]:
                                    st.image(result["image"], caption=f"Variation {i+1}", use_column_width=True)
                                    st.caption(result["filename"])

                        # Show generation details
                        with st.expander("📋 Generation Details", expanded=False):
                            for i, result in enumerate(results):
                                st.write(f"**Image {i+1}:**")
                                st.json({
                                    "enhanced_prompt": result["metadata"]["enhanced_prompt"],
                                    "data_type": result["metadata"]["data_type"],
                                    "generation_params": result["metadata"]["generation_params"]
                                })
                                if result["metadata"].get("data_features"):
                                    st.write("**Data Insights:**")
                                    st.json({
                                        "complexity_score": result["metadata"]["data_features"]["complexity_score"],
                                        "variability_score": result["metadata"]["data_features"]["variability_score"],
                                        "pattern_strength": result["metadata"]["data_features"]["pattern_strength"]
                                    })

                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    st.exception(e)

    # Clean up temporary files
    if csv_path and os.path.exists(csv_path):
        try:
            os.unlink(csv_path)
        except:
            pass

elif page == "🌀 Phase 2.C: Emotion-to-Image":
    st.markdown('<h2 class="phase-header">🌀 Phase 2.C: Emotional/Contextual Input to Image Generation</h2>', unsafe_allow_html=True)

    st.markdown("""
    **Transform emotions into stunning visual art!** Express your feelings through preset emotions,
    custom descriptions, or emojis, and watch as AI creates artwork that resonates with your emotional state.
    """)

    # Emotion-to-image interface
    col1, col2 = st.columns([2, 1])

    with col1:
        text_prompt = st.text_input(
            "Text Prompt",
            value="A tranquil landscape reflecting inner peace",
            help="Describe what you want to see in the image"
        )

        style = st.text_input(
            "Art Style",
            value="digital painting",
            help="e.g., oil painting, watercolor, abstract, impressionist"
        )

    with col2:
        num_images = st.number_input(
            "Number of Images",
            min_value=1, max_value=4, value=1, step=1
        )

        num_steps = st.slider(
            "Quality Steps",
            min_value=10, max_value=50, value=30, step=5,
            help="More steps = higher quality but slower generation"
        )

        enhancement_strength = st.slider(
            "Emotion Strength",
            min_value=0.0, max_value=1.0, value=0.7, step=0.1,
            help="How strongly to apply emotional conditioning"
        )

    # Emotion input section
    st.header("🌈 Emotional Input")

    emotion_method = st.radio(
        "Choose how to express your emotion:",
        ["Preset Emotions", "Custom Emotion/Emoji", "Descriptive Text"],
        help="Select your preferred method for emotional input"
    )

    emotion_input = ""
    emotion_type = "auto"
    contextual_text = ""

    if emotion_method == "Preset Emotions":
        # Preset emotions
        preset_emotions = [
            "joyful", "happy", "sad", "melancholic", "romantic", "peaceful",
            "mysterious", "energetic", "angry", "fearful", "surprised", "loving",
            "serene", "dramatic", "whimsical", "nostalgic", "uplifting"
        ]
        emotion_input = st.selectbox("Select Emotion:", preset_emotions)
        emotion_type = "preset"

    elif emotion_method == "Custom Emotion/Emoji":
        emotion_input = st.text_input(
            "Enter emotion or emoji:",
            placeholder="e.g., 'contemplative', '🤩', 'bittersweet'",
            help="Type any emotion word or use emojis"
        )
        emotion_type = "custom"

        # Quick emoji buttons
        st.write("**Quick Emoji Selection:**")
        emoji_cols = st.columns(6)
        quick_emojis = ["😊", "😢", "😡", "😱", "🤩", "❤️"]

        for i, emoji in enumerate(quick_emojis):
            with emoji_cols[i]:
                if st.button(emoji, key=f"emotion_emoji_{i}"):
                    emotion_input = emoji

    else:  # Descriptive Text
        contextual_text = st.text_area(
            "Describe your feeling:",
            placeholder="e.g., 'I feel hopeful after the rain'",
            height=100,
            help="Describe your emotional state in your own words"
        )
        emotion_input = contextual_text
        emotion_type = "text"

    # Emotion analysis
    emotion_analysis = None

    if emotion_input.strip():
        with st.spinner("Analyzing emotional context... 🧠"):
            try:
                from src.generators.compi_phase2c_emotion_to_image import CompIPhase2CEmotionToImage

                generator = CompIPhase2CEmotionToImage()
                emotion_analysis = generator.analyze_emotion(emotion_input, emotion_type, contextual_text)

                st.success("✅ Emotion analysis complete!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Primary Emotion", emotion_analysis.primary_emotion.value.title())
                    st.metric("Confidence", f"{emotion_analysis.emotion_confidence:.2f}")
                with col2:
                    st.metric("Sentiment", f"{emotion_analysis.sentiment_polarity:.2f}")
                    st.metric("Intensity", emotion_analysis.intensity_level.title())
                with col3:
                    if emotion_analysis.artistic_descriptors:
                        st.info(f"🎭 **Style:** {', '.join(emotion_analysis.artistic_descriptors[:2])}")
                    if emotion_analysis.color_palette:
                        st.info(f"🎨 **Colors:** {len(emotion_analysis.color_palette)} palette colors")

            except Exception as e:
                st.error(f"Error analyzing emotion: {e}")
                emotion_analysis = None

    # Prompt preview
    st.header("📝 Enhanced Prompt Preview")

    enhanced_prompt = text_prompt
    if style:
        enhanced_prompt += f", {style}"

    if emotion_analysis and enhancement_strength > 0:
        # Add emotional enhancement preview
        if emotion_analysis.artistic_descriptors:
            enhanced_prompt += f", {', '.join(emotion_analysis.artistic_descriptors[:2])}"
        if emotion_analysis.mood_modifiers:
            enhanced_prompt += f", with a {', '.join(emotion_analysis.mood_modifiers[:2])} atmosphere"

    st.markdown(f"**Enhanced Prompt:** `{enhanced_prompt}`")

    # Generation section
    st.header("🎨 Generate Emotional Art")

    if st.button("🚀 Generate Emotional Art", type="primary", use_container_width=True):
        if not text_prompt.strip():
            st.error("Please enter a text prompt!")
        elif not emotion_input.strip():
            st.error("Please provide emotional input!")
        else:
            with st.spinner(f"Generating {num_images} image(s)... This may take a few minutes."):
                try:
                    from src.generators.compi_phase2c_emotion_to_image import CompIPhase2CEmotionToImage

                    # Initialize generator
                    generator = CompIPhase2CEmotionToImage(output_dir="outputs")

                    # Generate images
                    results = generator.generate_image(
                        text_prompt=text_prompt,
                        style=style,
                        emotion_input=emotion_input,
                        emotion_type=emotion_type,
                        contextual_text=contextual_text,
                        enhancement_strength=enhancement_strength,
                        num_images=num_images,
                        num_inference_steps=num_steps,
                        guidance_scale=7.5
                    )

                    if results:
                        st.success(f"✅ Generated {len(results)} image(s) successfully!")

                        # Display results
                        if len(results) == 1:
                            result = results[0]
                            col1, col2 = st.columns([2, 1])

                            with col1:
                                st.image(result["image"], caption=result["filename"], use_column_width=True)
                                st.info(f"💾 Saved as: `{result['filename']}.png`")

                            with col2:
                                if result["emotion_analysis"]:
                                    st.markdown("**🎭 Emotion Summary:**")
                                    st.write(f"Primary: {result['emotion_analysis'].primary_emotion.value.title()}")
                                    st.write(f"Intensity: {result['emotion_analysis'].intensity_level.title()}")
                                    st.write(f"Sentiment: {result['emotion_analysis'].sentiment_polarity:.2f}")

                                    # Show color palette
                                    if result["emotion_analysis"].color_palette:
                                        st.markdown("**🎨 Color Palette:**")
                                        palette_html = ""
                                        for color in result["emotion_analysis"].color_palette:
                                            palette_html += f'<div style="display:inline-block; width:25px; height:25px; background-color:{color}; margin:2px; border:1px solid #ddd;"></div>'
                                        st.markdown(palette_html, unsafe_allow_html=True)
                        else:
                            # Multiple images
                            cols = st.columns(min(len(results), 3))
                            for i, result in enumerate(results):
                                with cols[i % len(cols)]:
                                    st.image(result["image"], caption=f"Variation {i+1}", use_column_width=True)
                                    st.caption(result["filename"])

                        # Show generation details
                        with st.expander("📋 Generation Details", expanded=False):
                            for i, result in enumerate(results):
                                st.write(f"**Image {i+1}:**")
                                st.json({
                                    "enhanced_prompt": result["metadata"]["enhanced_prompt"],
                                    "emotion_input": result["metadata"]["emotion_input"],
                                    "enhancement_strength": result["metadata"]["enhancement_strength"],
                                    "generation_params": result["metadata"]["generation_params"]
                                })
                                if result["metadata"].get("emotion_analysis"):
                                    st.write("**Emotion Analysis:**")
                                    st.json({
                                        "primary_emotion": result["metadata"]["emotion_analysis"]["primary_emotion"],
                                        "emotion_confidence": result["metadata"]["emotion_analysis"]["emotion_confidence"],
                                        "sentiment_polarity": result["metadata"]["emotion_analysis"]["sentiment_polarity"],
                                        "intensity_level": result["metadata"]["emotion_analysis"]["intensity_level"]
                                    })

                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    st.exception(e)

elif page == "🌎 Phase 2.D: Real-Time Data-to-Image":
    st.markdown('<h2 class="phase-header">🌎 Phase 2.D: Real-Time Data Feeds to Image Generation</h2>', unsafe_allow_html=True)

    st.markdown("""
    **Connect your art to the pulse of the world!** Incorporate live weather data, breaking news,
    and market information to create artwork that captures the current moment in time.
    """)

    # Real-time data-to-image interface
    col1, col2 = st.columns([2, 1])

    with col1:
        text_prompt = st.text_input(
            "Text Prompt",
            value="A landscape that reflects the world's current mood",
            help="Describe what you want to see in the image"
        )

        style = st.text_input(
            "Art Style",
            value="impressionist digital painting",
            help="e.g., photorealistic, abstract, oil painting, cyberpunk"
        )

        mood = st.text_input(
            "Base Mood/Atmosphere",
            value="contemplative, dynamic",
            help="Base mood that will be enhanced by real-time data"
        )

    with col2:
        num_images = st.number_input(
            "Number of Images",
            min_value=1, max_value=4, value=1, step=1
        )

        num_steps = st.slider(
            "Quality Steps",
            min_value=10, max_value=50, value=30, step=5,
            help="More steps = higher quality but slower generation"
        )

        context_strength = st.slider(
            "Real-Time Context Strength",
            min_value=0.0, max_value=1.0, value=0.7, step=0.1,
            help="How strongly to apply real-time data context"
        )

    # Real-time data source configuration
    st.header("🌐 Real-Time Data Sources")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🌤️ Weather Data")
        include_weather = st.checkbox("Include Weather Data", value=False, key="rt_weather")

        if include_weather:
            weather_city = st.text_input(
                "City",
                value="New York",
                help="City for weather data",
                key="rt_weather_city"
            )

            weather_api_key = st.text_input(
                "OpenWeatherMap API Key (Optional)",
                type="password",
                help="Leave blank to use demo key",
                key="rt_weather_key"
            )

    with col2:
        st.subheader("📰 News Data")
        include_news = st.checkbox("Include News Headlines", value=False, key="rt_news")

        if include_news:
            news_category = st.selectbox(
                "News Category",
                ["general", "technology", "science", "world", "business"],
                help="Category of news to fetch",
                key="rt_news_category"
            )

            max_news = st.slider(
                "Max Headlines",
                min_value=1, max_value=10, value=3, step=1,
                help="Maximum number of headlines to include",
                key="rt_max_news"
            )

    with col3:
        st.subheader("💹 Financial Data")
        include_financial = st.checkbox("Include Market Data", value=False, key="rt_financial")

        if include_financial:
            st.info("📊 **Available Data:**\n- Bitcoin price\n- USD exchange rates\n- Market indicators")

    # Real-time context preview
    if include_weather or include_news or include_financial:
        st.header("📊 Real-Time Context Preview")

        if st.button("🔄 Fetch Current Data", key="rt_fetch_data"):
            with st.spinner("Fetching real-time data..."):
                try:
                    from src.generators.compi_phase2d_realtime_to_image import CompIPhase2DRealTimeToImage

                    generator = CompIPhase2DRealTimeToImage()
                    context = generator.fetch_realtime_context(
                        include_weather=include_weather,
                        weather_city=weather_city if include_weather else "New York",
                        weather_api_key=weather_api_key if include_weather and weather_api_key else None,
                        include_news=include_news,
                        news_category=news_category if include_news else "general",
                        max_news=max_news if include_news else 3,
                        include_financial=include_financial
                    )

                    if context.data_points:
                        st.success("✅ Real-time data fetched successfully!")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**📝 Summary:** {context.summary}")
                            if context.mood_indicators:
                                st.write(f"**🎭 Mood:** {', '.join(context.mood_indicators)}")

                        with col2:
                            if context.key_themes:
                                st.write(f"**🎯 Themes:** {', '.join(context.key_themes)}")
                            st.write(f"**🎨 Inspiration:** {context.artistic_inspiration}")

                        # Store context for generation
                        st.session_state['rt_context'] = context
                    else:
                        st.warning("⚠️ No real-time data could be fetched")

                except Exception as e:
                    st.error(f"❌ Error fetching real-time data: {e}")

    # Prompt preview
    st.header("📝 Enhanced Prompt Preview")

    enhanced_prompt = text_prompt
    if style:
        enhanced_prompt += f", {style}"
    if mood:
        enhanced_prompt += f", {mood}"

    # Add real-time context preview if available
    if 'rt_context' in st.session_state and context_strength > 0:
        context = st.session_state['rt_context']
        if context_strength > 0.7:
            enhanced_prompt += f", {context.artistic_inspiration}"
            if context.mood_indicators:
                mood_text = ", ".join(context.mood_indicators[:2])
                enhanced_prompt += f", with {mood_text} influences"
        elif context_strength > 0.4:
            enhanced_prompt += f", {context.artistic_inspiration}"
        else:
            if context.key_themes:
                theme = context.key_themes[0]
                enhanced_prompt += f", inspired by {theme}"

    st.markdown(f"**Enhanced Prompt:** `{enhanced_prompt}`")

    # Show active data sources
    data_sources = []
    if include_weather:
        data_sources.append("🌤️ Weather")
    if include_news:
        data_sources.append("📰 News")
    if include_financial:
        data_sources.append("💹 Financial")

    if data_sources:
        st.info(f"**📡 Active Data Sources:** {' | '.join(data_sources)}")

    # Generation section
    st.header("🎨 Generate Real-Time Art")

    if st.button("🚀 Generate Real-Time Art", type="primary", use_container_width=True, key="rt_generate"):
        if not text_prompt.strip():
            st.error("Please enter a text prompt!")
        elif not (include_weather or include_news or include_financial):
            st.error("Please enable at least one real-time data source!")
        else:
            with st.spinner(f"Generating {num_images} image(s)... This may take a few minutes."):
                try:
                    from src.generators.compi_phase2d_realtime_to_image import CompIPhase2DRealTimeToImage

                    # Initialize generator
                    generator = CompIPhase2DRealTimeToImage(output_dir="outputs")

                    # Generate images
                    results = generator.generate_image(
                        text_prompt=text_prompt,
                        style=style,
                        mood=mood,
                        include_weather=include_weather,
                        weather_city=weather_city if include_weather else "New York",
                        weather_api_key=weather_api_key if include_weather and weather_api_key else None,
                        include_news=include_news,
                        news_category=news_category if include_news else "general",
                        max_news=max_news if include_news else 3,
                        include_financial=include_financial,
                        context_strength=context_strength,
                        num_images=num_images,
                        num_inference_steps=num_steps,
                        guidance_scale=7.5
                    )

                    if results:
                        st.success(f"✅ Generated {len(results)} image(s) successfully!")

                        # Display results
                        if len(results) == 1:
                            result = results[0]
                            col1, col2 = st.columns([2, 1])

                            with col1:
                                st.image(result["image"], caption=result["filename"], use_column_width=True)
                                st.info(f"💾 Saved as: `{result['filename']}.png`")

                            with col2:
                                if result["realtime_context"]:
                                    st.markdown("**🌐 Real-Time Context:**")
                                    st.write(f"📝 {result['realtime_context'].summary}")

                                    if result["realtime_context"].mood_indicators:
                                        st.write(f"🎭 **Mood:** {', '.join(result['realtime_context'].mood_indicators)}")

                                    if result["realtime_context"].key_themes:
                                        st.write(f"🎯 **Themes:** {', '.join(result['realtime_context'].key_themes)}")

                                    st.write(f"🎨 **Inspiration:** {result['realtime_context'].artistic_inspiration}")
                        else:
                            # Multiple images
                            cols = st.columns(min(len(results), 3))
                            for i, result in enumerate(results):
                                with cols[i % len(cols)]:
                                    st.image(result["image"], caption=f"Variation {i+1}", use_column_width=True)
                                    st.caption(result["filename"])

                        # Show generation details
                        with st.expander("📋 Generation Details", expanded=False):
                            for i, result in enumerate(results):
                                st.write(f"**Image {i+1}:**")
                                st.json({
                                    "enhanced_prompt": result["metadata"]["enhanced_prompt"],
                                    "data_sources": result["metadata"]["data_sources"],
                                    "context_strength": result["metadata"]["context_strength"],
                                    "generation_params": result["metadata"]["generation_params"]
                                })

                                if result["metadata"].get("realtime_context"):
                                    st.write("**Real-Time Context:**")
                                    context_summary = {
                                        "summary": result["metadata"]["realtime_context"]["summary"],
                                        "mood_indicators": result["metadata"]["realtime_context"]["mood_indicators"],
                                        "key_themes": result["metadata"]["realtime_context"]["key_themes"],
                                        "temporal_context": result["metadata"]["realtime_context"]["temporal_context"]
                                    }
                                    st.json(context_summary)

                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    st.exception(e)

elif page == "🖼️ Image Gallery":
    st.markdown('<h2 class="phase-header">🖼️ Image Gallery & Management</h2>', unsafe_allow_html=True)
    
    # Gallery interface
    try:
        outputs_dir = Path("outputs")
        if not outputs_dir.exists():
            st.info("No images found. Generate some images first!")
        else:
            # Get all images
            image_files = list(outputs_dir.glob("*.png"))
            
            if not image_files:
                st.info("No images found in the outputs directory.")
            else:
                st.write(f"Found {len(image_files)} images")
                
                # Sorting options
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    sort_by = st.selectbox("Sort by", ["Date (newest)", "Date (oldest)", "Name"])
                with col2:
                    images_per_row = st.selectbox("Images per row", [2, 3, 4, 5], index=1)
                
                # Sort images
                if sort_by == "Date (newest)":
                    image_files = sorted(image_files, key=lambda x: x.stat().st_mtime, reverse=True)
                elif sort_by == "Date (oldest)":
                    image_files = sorted(image_files, key=lambda x: x.stat().st_mtime)
                else:
                    image_files = sorted(image_files, key=lambda x: x.name)
                
                # Pagination
                images_per_page = 20
                total_pages = (len(image_files) + images_per_page - 1) // images_per_page
                
                if total_pages > 1:
                    page_num = st.selectbox("Page", range(1, total_pages + 1))
                    start_idx = (page_num - 1) * images_per_page
                    end_idx = start_idx + images_per_page
                    current_images = image_files[start_idx:end_idx]
                else:
                    current_images = image_files
                
                # Display images in grid
                from PIL import Image
                
                for i in range(0, len(current_images), images_per_row):
                    cols = st.columns(images_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(current_images):
                            img_path = current_images[i + j]
                            with col:
                                try:
                                    img = Image.open(img_path)
                                    st.image(img, caption=img_path.stem[:20] + "...", use_container_width=True)
                                    
                                    # Image info
                                    file_size = img_path.stat().st_size / 1024  # KB
                                    st.caption(f"Size: {file_size:.1f} KB")
                                    
                                    # Check for metadata
                                    metadata_path = img_path.with_suffix('.json')
                                    if metadata_path.exists():
                                        if st.button(f"📋 Info", key=f"info_{img_path.stem}"):
                                            import json
                                            with open(metadata_path) as f:
                                                metadata = json.load(f)
                                            st.json(metadata)
                                
                                except Exception as e:
                                    st.error(f"Error loading {img_path.name}: {e}")
    
    except Exception as e:
        st.error(f"Error loading gallery: {e}")

elif page == "📊 Analytics & Evaluation":
    st.markdown('<h2 class="phase-header">📊 Analytics & Evaluation</h2>', unsafe_allow_html=True)
    
    # Analytics interface
    st.markdown("### 📈 Generation Statistics")
    
    try:
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            image_files = list(outputs_dir.glob("*.png"))
            metadata_files = list(outputs_dir.glob("*.json"))
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Images", len(image_files))
            with col2:
                st.metric("With Metadata", len(metadata_files))
            with col3:
                audio_images = len([f for f in image_files if "AUDIO" in f.name])
                st.metric("Audio-Generated", audio_images)
            with col4:
                text_images = len(image_files) - audio_images
                st.metric("Text-Only", text_images)
            
            # Recent activity chart
            if image_files:
                st.markdown("### 📅 Generation Timeline")
                
                import pandas as pd
                from datetime import datetime
                
                # Extract dates from filenames or file modification times
                dates = []
                for img_file in image_files:
                    try:
                        # Try to extract date from filename
                        parts = img_file.stem.split('_')
                        date_part = None
                        for part in parts:
                            if len(part) == 8 and part.isdigit():
                                date_part = part
                                break
                        
                        if date_part:
                            date = datetime.strptime(date_part, "%Y%m%d").date()
                        else:
                            date = datetime.fromtimestamp(img_file.stat().st_mtime).date()
                        
                        dates.append(date)
                    except:
                        dates.append(datetime.fromtimestamp(img_file.stat().st_mtime).date())
                
                df = pd.DataFrame({'date': dates})
                daily_counts = df.groupby('date').size().reset_index(name='count')
                
                st.line_chart(daily_counts.set_index('date'))
        
        else:
            st.info("No data available. Generate some images first!")
    
    except Exception as e:
        st.error(f"Error loading analytics: {e}")

elif page == "📚 Documentation":
    st.markdown('<h2 class="phase-header">📚 Documentation & Guides</h2>', unsafe_allow_html=True)
    
    # Documentation interface
    doc_section = st.selectbox(
        "Choose documentation:",
        [
            "🚀 Quick Start Guide",
            "🎨 Phase 1 Guide", 
            "🎵 Phase 2.A Guide",
            "📖 API Reference",
            "🔧 Troubleshooting",
            "📋 Project Structure"
        ]
    )
    
    if doc_section == "🚀 Quick Start Guide":
        st.markdown("""
        ## Quick Start Guide
        
        ### 1. Basic Text-to-Image Generation
        1. Go to **Phase 1: Text-to-Image**
        2. Enter your prompt (e.g., "A magical forest at sunset")
        3. Choose style and mood
        4. Click **Generate Images**
        
        ### 2. Audio-to-Image Generation
        1. Go to **Phase 2.A: Audio-to-Image**
        2. Enter your text prompt
        3. Upload an audio file (MP3, WAV, etc.)
        4. Watch the audio analysis
        5. Click **Generate Art**
        
        ### 3. View Your Creations
        - Check the **Image Gallery** to see all generated images
        - Use **Analytics** to track your generation statistics
        """)
    
    elif doc_section == "🎵 Phase 2.A Guide":
        try:
            with open("PHASE2A_AUDIO_TO_IMAGE_GUIDE.md", "r") as f:
                guide_content = f.read()
            st.markdown(guide_content)
        except:
            st.error("Phase 2.A guide not found")
    
    else:
        st.info(f"Documentation for {doc_section} coming soon!")

elif page == "⚙️ Settings & Tools":
    st.markdown('<h2 class="phase-header">⚙️ Settings & Tools</h2>', unsafe_allow_html=True)
    
    # Settings interface
    st.markdown("### 🔧 System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Model Settings")
        
        model_name = st.selectbox(
            "Stable Diffusion Model",
            [
                "runwayml/stable-diffusion-v1-5",
                "stabilityai/stable-diffusion-2-1",
                "stabilityai/stable-diffusion-xl-base-1.0"
            ]
        )
        
        whisper_model = st.selectbox(
            "Whisper Model Size",
            ["tiny", "base", "small", "medium", "large"]
        )
        
        default_steps = st.slider("Default Generation Steps", 10, 50, 30)
    
    with col2:
        st.markdown("#### Output Settings")
        
        output_format = st.selectbox("Image Format", ["PNG", "JPEG"])
        save_metadata = st.checkbox("Save Metadata", value=True)
        auto_cleanup = st.checkbox("Auto-cleanup temp files", value=True)
    
    st.markdown("### 🛠️ Utility Tools")
    
    tool_col1, tool_col2, tool_col3 = st.columns(3)
    
    with tool_col1:
        if st.button("🧹 Clean Temp Files", use_container_width=True):
            # Clean temporary files
            import glob
            temp_files = glob.glob("temp_*.wav") + glob.glob("demo_*.wav")
            cleaned = 0
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                    cleaned += 1
                except:
                    pass
            st.success(f"Cleaned {cleaned} temporary files")
    
    with tool_col2:
        if st.button("📊 System Info", use_container_width=True):
            st.info(f"""
            **System Information:**
            - Device: {device.upper()}
            - Python: {sys.version.split()[0]}
            - PyTorch: {torch.__version__}
            - CUDA Available: {torch.cuda.is_available()}
            """)
    
    with tool_col3:
        if st.button("🔄 Restart Models", use_container_width=True):
            # Clear model cache
            if 'generator' in st.session_state:
                del st.session_state.generator
            st.success("Model cache cleared. Models will reload on next use.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <strong>CompI: Complete Multimodal AI Art Platform</strong><br>
    Combining Text, Audio, and Advanced AI for Revolutionary Art Generation<br>
    🎨 Built with Stable Diffusion • 🎵 Powered by Whisper • ⚡ Accelerated by PyTorch
</div>
""", unsafe_allow_html=True)

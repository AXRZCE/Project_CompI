"""
CompI Phase 2.A: Audio-to-Image Streamlit UI

Interactive web interface for multimodal AI art generation combining:
- Text prompts with style and mood
- Audio analysis and captioning
- Real-time feature visualization
- Comprehensive generation controls
"""

import os
import sys
import torch
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.generators.compi_phase2a_audio_to_image import CompIPhase2AAudioToImage
from src.utils.audio_utils import AudioFeatures

# ------------------------ CONFIGURATION ------------------------

st.set_page_config(
    page_title="CompI Phase 2.A: Audio-to-Art", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .audio-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------ INITIALIZATION ------------------------

@st.cache_resource(show_spinner=True)
def load_generator():
    """Load and cache the CompI Phase 2.A generator"""
    return CompIPhase2AAudioToImage()

# Display header
st.markdown('<h1 class="main-header">🎶 CompI Phase 2.A: Audio-to-Art Generator</h1>', unsafe_allow_html=True)

st.markdown("""
**Welcome to CompI Phase 2.A!** Upload an audio file (music, voice, ambient sound) and watch as AI transforms 
its essence into unique visual art. Combine traditional text prompts with the emotional and rhythmic qualities 
of sound for truly multimodal creativity.
""")

# Device information
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"🖥️ Running on: {device.upper()}")
if device == "cuda":
    try:
        gpu_name = torch.cuda.get_device_name(0)
        st.sidebar.success(f"🚀 GPU: {gpu_name}")
    except:
        pass

# Load generator
with st.spinner("Loading AI models... (this may take a moment on first run)"):
    generator = load_generator()

# ------------------------ INPUT SECTION ------------------------

st.header("🎨 Generation Settings")

col1, col2 = st.columns([2, 1])

with col1:
    # Text inputs
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
    # Generation parameters
    st.subheader("Generation Parameters")
    
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

# ------------------------ AUDIO SECTION ------------------------

st.header("🎵 Audio Input")

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["mp3", "wav", "flac", "m4a", "ogg"],
    help="Upload music, voice, or ambient sound (recommended: under 60 seconds for faster processing)"
)

audio_features = None
audio_caption = ""

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_audio_path = f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # Display audio player
    st.audio(uploaded_file, format='audio/wav')
    
    # Analyze audio
    with st.spinner("Analyzing audio... 🎵"):
        try:
            audio_features, audio_caption = generator.analyze_audio(temp_audio_path)
            
            # Display audio information
            st.markdown('<div class="audio-info">', unsafe_allow_html=True)
            st.success("✅ Audio analysis complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duration", f"{audio_features.duration:.1f}s")
                st.metric("Tempo", f"{audio_features.tempo:.1f} BPM")
            with col2:
                st.metric("Energy", f"{audio_features.energy:.4f}")
                st.metric("Brightness", f"{audio_features.spectral_centroid:.0f} Hz")
            with col3:
                st.metric("Rhythm", f"{audio_features.zero_crossing_rate:.3f}")
                st.metric("Sample Rate", f"{audio_features.sample_rate} Hz")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display audio caption
            if audio_caption:
                st.info(f"🎙️ **Audio Caption:** \"{audio_caption}\"")
            
        except Exception as e:
            st.error(f"Error analyzing audio: {e}")
            audio_features = None
    
    # Clean up temp file
    try:
        os.remove(temp_audio_path)
    except:
        pass

# ------------------------ AUDIO VISUALIZATION ------------------------

if audio_features is not None:
    st.header("📊 Audio Analysis Visualization")
    
    # Create feature visualization
    with st.expander("🔍 Detailed Audio Features", expanded=False):
        
        # MFCC visualization
        fig_mfcc = go.Figure()
        fig_mfcc.add_trace(go.Bar(
            x=[f"MFCC {i+1}" for i in range(len(audio_features.mfcc_mean))],
            y=audio_features.mfcc_mean,
            name="MFCC Coefficients",
            marker_color='lightblue'
        ))
        fig_mfcc.update_layout(
            title="Mel-Frequency Cepstral Coefficients (Timbre)",
            xaxis_title="MFCC Index",
            yaxis_title="Coefficient Value"
        )
        st.plotly_chart(fig_mfcc, use_container_width=True)
        
        # Chroma visualization
        fig_chroma = go.Figure()
        chroma_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        fig_chroma.add_trace(go.Bar(
            x=chroma_notes,
            y=audio_features.chroma_mean,
            name="Chroma Features",
            marker_color='lightgreen'
        ))
        fig_chroma.update_layout(
            title="Chroma Features (Harmonic Content)",
            xaxis_title="Musical Note",
            yaxis_title="Chroma Intensity"
        )
        st.plotly_chart(fig_chroma, use_container_width=True)
        
        # Audio tags
        audio_tags = generator.prompt_fusion.generate_audio_tags(audio_features)
        st.write("**Generated Audio Tags:**", ", ".join(audio_tags))

# ------------------------ PROMPT PREVIEW ------------------------

if audio_features is not None:
    enhanced_prompt = generator.prompt_fusion.fuse_prompt_with_audio(
        text_prompt, style, mood, audio_features, audio_caption
    )
else:
    enhanced_prompt = text_prompt
    if style:
        enhanced_prompt += f", {style}"
    if mood:
        enhanced_prompt += f", {mood}"

st.header("📝 Final Prompt Preview")
st.markdown(f"**Enhanced Prompt:** `{enhanced_prompt}`")

# ------------------------ GENERATION ------------------------

st.header("🎨 Generate Images")

if st.button("🚀 Generate Art", type="primary", use_container_width=True):
    if not text_prompt.strip():
        st.error("Please enter a text prompt!")
    else:
        with st.spinner(f"Generating {num_images} image(s)... This may take a few minutes."):
            try:
                # Handle audio path for generation
                audio_path_for_generation = None
                if uploaded_file is not None:
                    # Re-save the uploaded file for generation
                    temp_audio_path_gen = f"temp_audio_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                    with open(temp_audio_path_gen, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    audio_path_for_generation = temp_audio_path_gen

                results = generator.generate_image(
                    text_prompt=text_prompt,
                    style=style,
                    mood=mood,
                    audio_path=audio_path_for_generation,
                    num_images=num_images,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale
                )

                # Clean up temp file
                if audio_path_for_generation:
                    try:
                        os.remove(audio_path_for_generation)
                    except:
                        pass
                
                st.success(f"✅ Generated {len(results)} image(s) successfully!")
                
                # Display results
                if len(results) == 1:
                    result = results[0]
                    st.image(result["image"], caption=result["filename"], use_column_width=True)
                    st.info(f"💾 Saved as: `{result['filename']}.png`")
                else:
                    cols = st.columns(min(len(results), 3))
                    for i, result in enumerate(results):
                        with cols[i % len(cols)]:
                            st.image(result["image"], caption=f"Variation {i+1}", use_column_width=True)
                            st.caption(result["filename"])
                
                # Show generation log
                with st.expander("📋 Generation Details", expanded=False):
                    for i, result in enumerate(results):
                        st.json(result["metadata"]["generation_params"])
                
            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.exception(e)

# ------------------------ FOOTER ------------------------

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <strong>CompI Phase 2.A — Audio-to-Art</strong><br>
    Multimodal AI Art Generation • Built with Stable Diffusion, Whisper & Streamlit<br>
    🎵 + 🎨 = ✨ Magic
</div>
""", unsafe_allow_html=True)

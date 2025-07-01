"""
CompI Phase 2.C: Emotional/Contextual Input Streamlit UI

Interactive web interface for emotion-driven AI art generation combining:
- Emotion selection and custom emotion input
- Sentiment analysis and mood detection
- Emoji and contextual text processing
- Color palette visualization based on emotions
- Comprehensive generation controls
"""

import os
import sys
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.generators.compi_phase2c_emotion_to_image import CompIPhase2CEmotionToImage
from src.utils.emotion_utils import EmotionAnalysis, EmotionCategory

# ------------------------ CONFIGURATION ------------------------

st.set_page_config(
    page_title="CompI Phase 2.C: Emotion-to-Art", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #9370DB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-info {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #9370DB;
    }
    .emotion-palette {
        display: flex;
        gap: 10px;
        margin: 10px 0;
    }
    .color-box {
        width: 40px;
        height: 40px;
        border-radius: 5px;
        border: 2px solid #ddd;
    }
    .sentiment-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #32CD32;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------ INITIALIZATION ------------------------

@st.cache_resource(show_spinner=True)
def load_generator():
    """Load and cache the CompI Phase 2.C generator"""
    return CompIPhase2CEmotionToImage()

# Display header
st.markdown('<h1 class="main-header">🌀 CompI Phase 2.C: Emotion-to-Art Generator</h1>', unsafe_allow_html=True)

st.markdown("""
**Welcome to CompI Phase 2.C!** Express your emotions and watch as AI transforms feelings into stunning visual art. 
Choose from preset emotions, describe your mood, or use emojis to create emotionally-rich artwork that resonates with your inner state.
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
        value="A tranquil landscape reflecting inner peace",
        help="Describe what you want to see in the image"
    )
    
    style = st.text_input(
        "Art Style", 
        value="digital painting",
        help="e.g., oil painting, watercolor, abstract, impressionist"
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
    
    enhancement_strength = st.slider(
        "Emotion Strength", 
        min_value=0.0, max_value=1.0, value=0.7, step=0.1,
        help="How strongly to apply emotional conditioning"
    )

# ------------------------ EMOTION INPUT SECTION ------------------------

st.header("🌈 Emotional Input")

# Emotion input method selection
emotion_method = st.radio(
    "Choose how to express your emotion:",
    ["Preset Emotions", "Custom Emotion/Emoji", "Descriptive Text"],
    help="Select your preferred method for emotional input"
)

emotion_input = ""
emotion_type = "auto"
contextual_text = ""

if emotion_method == "Preset Emotions":
    st.subheader("🎭 Select a Preset Emotion")
    
    # Preset emotions organized by category
    preset_emotions = {
        "Joy & Happiness": ["joyful", "happy", "ecstatic", "cheerful", "uplifting", "energetic"],
        "Sadness & Melancholy": ["sad", "melancholic", "nostalgic", "wistful", "somber"],
        "Love & Romance": ["romantic", "loving", "passionate", "tender", "affectionate"],
        "Peace & Serenity": ["peaceful", "serene", "tranquil", "calm", "harmonious"],
        "Mystery & Drama": ["mysterious", "dramatic", "enigmatic", "suspenseful"],
        "Energy & Power": ["powerful", "dynamic", "intense", "bold", "fierce"],
        "Fear & Anxiety": ["fearful", "anxious", "nervous", "worried", "tense"],
        "Anger & Frustration": ["angry", "frustrated", "furious", "irritated"],
        "Wonder & Surprise": ["surprised", "amazed", "astonished", "wonderstruck"],
        "Whimsy & Playfulness": ["whimsical", "playful", "quirky", "lighthearted"]
    }
    
    selected_category = st.selectbox("Emotion Category:", list(preset_emotions.keys()))
    emotion_input = st.selectbox("Specific Emotion:", preset_emotions[selected_category])
    emotion_type = "preset"

elif emotion_method == "Custom Emotion/Emoji":
    st.subheader("✨ Custom Emotion or Emoji")
    
    col1, col2 = st.columns(2)
    
    with col1:
        emotion_input = st.text_input(
            "Enter emotion or emoji:",
            placeholder="e.g., 'contemplative', '🤩', 'bittersweet'",
            help="Type any emotion word or use emojis"
        )
    
    with col2:
        # Quick emoji buttons
        st.write("**Quick Emoji Selection:**")
        emoji_cols = st.columns(6)
        quick_emojis = ["😊", "😢", "😡", "😱", "🤩", "❤️", "🌟", "🌙", "🔥", "💫", "🌈", "⚡"]
        
        for i, emoji in enumerate(quick_emojis):
            with emoji_cols[i % 6]:
                if st.button(emoji, key=f"emoji_{i}"):
                    emotion_input = emoji
    
    emotion_type = "custom"

else:  # Descriptive Text
    st.subheader("📝 Describe Your Feeling")
    
    contextual_text = st.text_area(
        "Describe your current mood or feeling:",
        placeholder="e.g., 'I feel hopeful after the rain', 'There's a sense of anticipation in the air', 'I'm overwhelmed by the beauty of nature'",
        height=100,
        help="Describe your emotional state in your own words"
    )
    
    emotion_input = contextual_text
    emotion_type = "text"

# ------------------------ EMOTION ANALYSIS ------------------------

emotion_analysis = None

if emotion_input.strip():
    with st.spinner("Analyzing emotional context... 🧠"):
        try:
            emotion_analysis = generator.analyze_emotion(emotion_input, emotion_type, contextual_text)
            
            # Display emotion analysis
            st.markdown('<div class="emotion-info">', unsafe_allow_html=True)
            st.success("✅ Emotion analysis complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Primary Emotion", emotion_analysis.primary_emotion.value.title())
                st.metric("Confidence", f"{emotion_analysis.emotion_confidence:.2f}")
            with col2:
                st.metric("Sentiment", f"{emotion_analysis.sentiment_polarity:.2f}")
                st.metric("Intensity", emotion_analysis.intensity_level.title())
            with col3:
                st.metric("Subjectivity", f"{emotion_analysis.sentiment_subjectivity:.2f}")
                st.metric("Detected Emojis", len(emotion_analysis.detected_emojis))
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display color palette
            if emotion_analysis.color_palette:
                st.subheader("🎨 Emotion Color Palette")
                
                # Create color visualization
                fig = go.Figure()
                
                for i, color in enumerate(emotion_analysis.color_palette):
                    fig.add_trace(go.Bar(
                        x=[f"Color {i+1}"],
                        y=[1],
                        marker_color=color,
                        name=color,
                        showlegend=False
                    ))
                
                fig.update_layout(
                    title="Colors Associated with Your Emotion",
                    xaxis_title="Color Palette",
                    yaxis_title="",
                    height=200,
                    showlegend=False
                )
                fig.update_yaxis(showticklabels=False)
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Display artistic descriptors
            if emotion_analysis.artistic_descriptors:
                st.info(f"🎭 **Artistic Style:** {', '.join(emotion_analysis.artistic_descriptors)}")
            
            # Display mood modifiers
            if emotion_analysis.mood_modifiers:
                st.info(f"🌟 **Mood Atmosphere:** {', '.join(emotion_analysis.mood_modifiers)}")
            
        except Exception as e:
            st.error(f"Error analyzing emotion: {e}")
            emotion_analysis = None

# ------------------------ DETAILED EMOTION ANALYSIS ------------------------

if emotion_analysis is not None:
    with st.expander("🔍 Detailed Emotion Analysis", expanded=False):

        # Emotion scores
        st.subheader("📊 Emotion Scores")

        emotion_scores_df = {
            'Emotion': list(emotion_analysis.emotion_scores.keys()),
            'Score': list(emotion_analysis.emotion_scores.values())
        }

        fig_scores = px.bar(
            emotion_scores_df,
            x='Emotion',
            y='Score',
            title="Detected Emotion Intensities",
            color='Score',
            color_continuous_scale='viridis'
        )
        fig_scores.update_layout(height=400)
        st.plotly_chart(fig_scores, use_container_width=True)

        # Keywords and emojis
        col1, col2 = st.columns(2)

        with col1:
            if emotion_analysis.emotion_keywords:
                st.write("**🔤 Detected Keywords:**")
                st.write(", ".join(emotion_analysis.emotion_keywords))

        with col2:
            if emotion_analysis.detected_emojis:
                st.write("**😀 Detected Emojis:**")
                st.write(" ".join(emotion_analysis.detected_emojis))

# ------------------------ PROMPT PREVIEW ------------------------

# Create enhanced prompt preview
enhanced_prompt = text_prompt
if style:
    enhanced_prompt += f", {style}"

if emotion_analysis and enhancement_strength > 0:
    # Show what the emotion enhancement will add
    emotion_enhancement = generator.prompt_enhancer.enhance_prompt_with_emotion(
        text_prompt, style, emotion_analysis, enhancement_strength
    )
    enhanced_prompt = emotion_enhancement

st.header("📝 Enhanced Prompt Preview")
st.markdown(f"**Enhanced Prompt:** `{enhanced_prompt}`")

if emotion_analysis:
    emotion_tags = generator.prompt_enhancer.generate_emotion_tags(emotion_analysis)
    st.info(f"🏷️ **Emotion Tags:** {', '.join(emotion_tags)}")

# ------------------------ GENERATION ------------------------

st.header("🎨 Generate Emotional Art")

# Advanced options
with st.expander("⚙️ Advanced Options", expanded=False):
    use_color_conditioning = st.checkbox(
        "Use Color Palette Conditioning",
        value=False,
        help="Add emotion-derived colors to the prompt"
    )

    batch_emotions = st.text_area(
        "Batch Emotions (one per line):",
        placeholder="happy\nsad\nmysterious\n🤩",
        help="Generate multiple images with different emotions"
    )

if st.button("🚀 Generate Emotional Art", type="primary", use_container_width=True):
    if not text_prompt.strip():
        st.error("Please enter a text prompt!")
    elif not emotion_input.strip() and not batch_emotions.strip():
        st.error("Please provide emotional input!")
    else:
        # Determine generation mode
        if batch_emotions.strip():
            # Batch processing
            emotions_list = [e.strip() for e in batch_emotions.split('\n') if e.strip()]

            with st.spinner(f"Generating art for {len(emotions_list)} emotions... This may take several minutes."):
                try:
                    results = generator.batch_process_emotions(
                        text_prompt=text_prompt,
                        style=style,
                        emotions=emotions_list,
                        enhancement_strength=enhancement_strength,
                        num_images=1,  # One per emotion for batch
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale
                    )

                    st.success(f"✅ Generated {len(results)} images successfully!")

                    # Display batch results
                    cols = st.columns(min(len(results), 4))
                    for i, result in enumerate(results):
                        with cols[i % len(cols)]:
                            st.image(result["image"], caption=f"Emotion: {emotions_list[i % len(emotions_list)]}", use_column_width=True)
                            st.caption(result["filename"])

                except Exception as e:
                    st.error(f"Batch generation failed: {e}")
                    st.exception(e)

        else:
            # Single emotion generation
            with st.spinner(f"Generating {num_images} image(s)... This may take a few minutes."):
                try:
                    if use_color_conditioning and emotion_analysis:
                        # Use color palette conditioning
                        results = generator.generate_emotion_palette_art(
                            text_prompt=text_prompt,
                            style=style,
                            emotion_input=emotion_input,
                            use_color_conditioning=True,
                            enhancement_strength=enhancement_strength,
                            num_images=num_images,
                            num_inference_steps=num_steps,
                            guidance_scale=guidance_scale
                        )
                    else:
                        # Standard generation
                        results = generator.generate_image(
                            text_prompt=text_prompt,
                            style=style,
                            emotion_input=emotion_input,
                            emotion_type=emotion_type,
                            contextual_text=contextual_text,
                            enhancement_strength=enhancement_strength,
                            num_images=num_images,
                            num_inference_steps=num_steps,
                            guidance_scale=guidance_scale
                        )

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
                                        palette_html += f'<div style="display:inline-block; width:30px; height:30px; background-color:{color}; margin:2px; border:1px solid #ddd;"></div>'
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
                            st.subheader(f"Image {i+1}")
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

# ------------------------ FOOTER ------------------------

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <strong>CompI Phase 2.C — Emotion-to-Art</strong><br>
    Transform Feelings into Visual Poetry • Built with Stable Diffusion, TextBlob & Streamlit<br>
    🌀 + 🎨 = ✨ Emotional Art
</div>
""", unsafe_allow_html=True)

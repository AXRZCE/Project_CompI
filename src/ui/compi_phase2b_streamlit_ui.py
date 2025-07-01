"""
CompI Phase 2.B: Data/Logic-to-Image Streamlit UI

Interactive web interface for data-driven AI art generation combining:
- CSV data upload and analysis
- Mathematical formula evaluation
- Data visualization and pattern recognition
- Poetic text generation from data insights
- Comprehensive generation controls
"""

import os
import sys
import torch
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
import tempfile

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.generators.compi_phase2b_data_to_image import CompIPhase2BDataToImage
from src.utils.data_utils import DataFeatures

# ------------------------ CONFIGURATION ------------------------

st.set_page_config(
    page_title="CompI Phase 2.B: Data-to-Art", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .data-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .formula-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #2E86AB;
        font-family: monospace;
    }
    .poetic-text {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #96CEB4;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------ INITIALIZATION ------------------------

@st.cache_resource(show_spinner=True)
def load_generator():
    """Load and cache the CompI Phase 2.B generator"""
    return CompIPhase2BDataToImage()

# Display header
st.markdown('<h1 class="main-header">📊 CompI Phase 2.B: Data/Logic-to-Art Generator</h1>', unsafe_allow_html=True)

st.markdown("""
**Welcome to CompI Phase 2.B!** Transform structured data (CSV files) or mathematical formulas into stunning AI-generated art. 
Watch as numbers, patterns, and equations inspire unique visual creations through intelligent data analysis and poetic interpretation.
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
        value="A visualization of nature's mathematical patterns",
        help="Describe what you want to see in the image"
    )
    
    style = st.text_input(
        "Art Style", 
        value="digital art",
        help="e.g., abstract, geometric, organic, minimalist, baroque"
    )
    
    mood = st.text_input(
        "Mood/Atmosphere", 
        value="harmonious, flowing",
        help="e.g., dynamic, peaceful, mysterious, energetic"
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

# ------------------------ DATA INPUT SECTION ------------------------

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
        help="Upload a CSV file with numeric data (time series, measurements, statistics, etc.)"
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
                df, data_features, poetic_description, data_visualization = generator.analyze_csv_data(csv_path)
                
                # Display data information
                st.markdown('<div class="data-info">', unsafe_allow_html=True)
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
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error analyzing data: {e}")
                data_features = None

else:  # Mathematical Formula
    st.subheader("🧮 Mathematical Formula")
    
    # Formula examples
    with st.expander("📚 Formula Examples", expanded=False):
        st.code("""
Examples of mathematical formulas you can use:

• Sine wave: np.sin(np.linspace(0, 4*np.pi, 100))
• Spiral: np.sin(np.linspace(0, 10, 100)) * np.exp(-np.linspace(0, 2, 100))
• Polynomial: np.linspace(-5, 5, 100)**3 - 3*np.linspace(-5, 5, 100)
• Exponential: np.exp(np.linspace(-2, 2, 100))
• Complex pattern: np.sin(x) + 0.5*np.cos(3*x) where x = np.linspace(0, 6*np.pi, 200)
• Random walk: np.cumsum(np.random.randn(100))

Note: Use 'x' as variable or define your own range with np.linspace()
        """)
    
    formula = st.text_area(
        "Enter Mathematical Formula",
        value="np.sin(np.linspace(0, 4*np.pi, 100)) * np.exp(-np.linspace(0, 1, 100))",
        height=100,
        help="Use Python/NumPy syntax. Examples: np.sin(x), np.exp(x), np.linspace(0, 10, 100)"
    )
    
    if formula.strip():
        with st.spinner("Evaluating formula... 🧮"):
            try:
                result_array, metadata, poetic_description, data_visualization = generator.evaluate_mathematical_formula(formula)
                
                # Display formula information
                st.markdown('<div class="formula-box">', unsafe_allow_html=True)
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
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error evaluating formula: {e}")
                formula = None

# ------------------------ DATA VISUALIZATION ------------------------

if data_visualization is not None:
    st.header("📊 Data Pattern Visualization")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(data_visualization, caption="Data Pattern Visualization", use_column_width=True)

    with col2:
        if poetic_description:
            st.markdown('<div class="poetic-text">', unsafe_allow_html=True)
            st.markdown("**🎭 Poetic Data Interpretation:**")
            st.write(poetic_description)
            st.markdown('</div>', unsafe_allow_html=True)

# ------------------------ DETAILED DATA ANALYSIS ------------------------

if data_features is not None:
    with st.expander("🔍 Detailed Data Analysis", expanded=False):

        # Statistical summary
        st.subheader("📈 Statistical Summary")

        if data_features.numeric_columns:
            stats_df = pd.DataFrame({
                'Column': data_features.numeric_columns,
                'Mean': [data_features.means[col] for col in data_features.numeric_columns],
                'Median': [data_features.medians[col] for col in data_features.numeric_columns],
                'Std Dev': [data_features.stds[col] for col in data_features.numeric_columns],
                'Min': [data_features.mins[col] for col in data_features.numeric_columns],
                'Max': [data_features.maxs[col] for col in data_features.numeric_columns],
                'Trend': [data_features.trends[col] for col in data_features.numeric_columns]
            })
            st.dataframe(stats_df)

        # Correlations
        if data_features.correlations:
            st.subheader("🔗 Strongest Correlations")
            for pair, corr in data_features.correlations.items():
                st.write(f"**{pair}**: {corr:.3f}")

# ------------------------ PROMPT PREVIEW ------------------------

enhanced_prompt = text_prompt
if style:
    enhanced_prompt += f", {style}"
if mood:
    enhanced_prompt += f", {mood}"
if poetic_description:
    enhanced_prompt += f", {poetic_description}"

st.header("📝 Final Prompt Preview")
st.markdown(f"**Enhanced Prompt:** `{enhanced_prompt}`")

# ------------------------ GENERATION ------------------------

st.header("🎨 Generate Images")

if st.button("🚀 Generate Art", type="primary", use_container_width=True):
    if not text_prompt.strip():
        st.error("Please enter a text prompt!")
    elif not csv_path and not formula:
        st.error("Please upload CSV data or enter a mathematical formula!")
    else:
        with st.spinner(f"Generating {num_images} image(s)... This may take a few minutes."):
            try:
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
                            if result["data_visualization_path"]:
                                st.caption(f"Data viz saved: `{os.path.basename(result['data_visualization_path'])}`")

                        if result["poetic_description"]:
                            st.markdown('<div class="poetic-text">', unsafe_allow_html=True)
                            st.markdown("**🎭 Data Poetry:**")
                            st.write(result["poetic_description"])
                            st.markdown('</div>', unsafe_allow_html=True)

                else:
                    # Multiple images
                    cols = st.columns(min(len(results), 3))
                    for i, result in enumerate(results):
                        with cols[i % len(cols)]:
                            st.image(result["image"], caption=f"Variation {i+1}", use_column_width=True)
                            st.caption(result["filename"])

                # Show generation log
                with st.expander("📋 Generation Details", expanded=False):
                    for i, result in enumerate(results):
                        st.subheader(f"Image {i+1}")
                        st.json(result["metadata"]["generation_params"])
                        if result["metadata"].get("data_features"):
                            st.write("**Data Features:**")
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

# ------------------------ FOOTER ------------------------

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <strong>CompI Phase 2.B — Data/Logic-to-Art</strong><br>
    Transform Numbers into Visual Poetry • Built with Stable Diffusion, Pandas & Streamlit<br>
    📊 + 🧮 + 🎨 = ✨ Mathematical Art
</div>
""", unsafe_allow_html=True)

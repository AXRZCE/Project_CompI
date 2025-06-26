import os
import sys
import torch
from datetime import datetime
from diffusers import StableDiffusionPipeline
from PIL import Image
import streamlit as st

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# ------------------------ 1. SETUP ------------------------

st.set_page_config(page_title="CompI - Text-to-Image AI", layout="wide")
st.title("🖼️ CompI: Text+Style+Mood to AI Art")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource(show_spinner=True)
def load_pipe():
    model_name = "runwayml/stable-diffusion-v1-5"
    def dummy_safety_checker(images, **kwargs): 
        return images, [False] * len(images)
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=dummy_safety_checker,
    )
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    return pipe

# Display device info
st.sidebar.info(f"🖥️ Running on: {DEVICE.upper()}")
if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    st.sidebar.success(f"🚀 GPU: {gpu_name}")

# Load model with progress
with st.spinner("Loading AI model... (this may take a moment on first run)"):
    pipe = load_pipe()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------ 2. UI INPUTS ------------------------

STYLES = [
    "digital art", "oil painting", "watercolor", "cyberpunk", "impressionist", "concept art",
    "anime", "photorealistic", "minimalist", "surrealism", "pixel art", "steampunk", "3d render"
]

MOODS = [
    "dreamy atmosphere", "dark and moody", "peaceful", "vibrant and energetic", "melancholic",
    "mysterious", "whimsical", "serene", "uplifting", "dramatic lighting", "retro"
]

# Sidebar for advanced settings
st.sidebar.header("⚙️ Generation Settings")
inference_steps = st.sidebar.slider("Inference Steps", 10, 50, 30, help="More steps = better quality, slower generation")
guidance_scale = st.sidebar.slider("Guidance Scale", 1.0, 20.0, 7.5, help="Higher = follows prompt more strictly")
height = st.sidebar.selectbox("Height", [512, 768], index=0)
width = st.sidebar.selectbox("Width", [512, 768], index=0)

# Main form
with st.form(key="prompt_form"):
    st.header("🎨 Create Your AI Art")
    
    main_prompt = st.text_input(
        "Describe your scene or subject:", 
        "A magical forest with glowing trees",
        help="Describe what you want to see in your image"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎭 Art Style")
        style = st.selectbox("Select an art style:", STYLES, index=0)
        custom_style = st.text_input("Or enter a custom style (optional):", "")
    
    with col2:
        st.subheader("🌟 Mood & Atmosphere")
        mood = st.selectbox("Select a mood/atmosphere:", MOODS, index=0)
        custom_mood = st.text_input("Or enter a custom mood (optional):", "")
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        negative_prompt = st.text_input(
            "Negative prompt (what to avoid):", 
            "blurry, low quality, distorted",
            help="Describe what you don't want in the image"
        )
        use_seed = st.checkbox("Use fixed seed for reproducibility")
        if use_seed:
            seed_value = st.number_input("Seed value:", min_value=0, max_value=2**32-1, value=42)
    
    num_variations = st.number_input("How many images?", min_value=1, max_value=5, value=1, step=1)
    
    submitted = st.form_submit_button("🎨 Generate Art", type="primary")

# Build full prompt
full_style = custom_style.strip() if custom_style.strip() else style
full_mood = custom_mood.strip() if custom_mood.strip() else mood

full_prompt = main_prompt.strip()
if full_style: full_prompt += f", {full_style}"
if full_mood:  full_prompt += f", {full_mood}"

# Show what will be generated
if main_prompt:
    st.info(f"**Full prompt:** {full_prompt}")
    if negative_prompt:
        st.info(f"**Negative prompt:** {negative_prompt}")

# ------------------------ 3. IMAGE GENERATION ------------------------

def save_image(img: Image.Image, prompt, style, mood, seed, idx):
    prompt_slug = "_".join(prompt.lower().split()[:5])
    style_slug = style.replace(" ", "")[:10]
    mood_slug = mood.replace(" ", "")[:10] if mood else "none"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{prompt_slug[:25]}_{style_slug}_{mood_slug}_{timestamp}_seed{seed}_v{idx+1}.png"
    fpath = os.path.join(OUTPUT_DIR, fname)
    img.save(fpath)
    return fpath

if submitted and main_prompt.strip():
    st.header("🎨 Generated Art")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    images = []
    filenames = []
    generation_logs = []
    
    for idx in range(num_variations):
        status_text.text(f"Generating image {idx+1} of {num_variations}...")
        progress_bar.progress((idx) / num_variations)
        
        # Set seed
        if use_seed:
            seed = seed_value + idx  # Increment for variations
        else:
            seed = torch.seed()
        
        generator = torch.manual_seed(seed) if DEVICE == "cpu" else torch.Generator(DEVICE).manual_seed(seed)
        
        try:
            with torch.autocast(DEVICE) if DEVICE == "cuda" else torch.no_grad():
                result = pipe(
                    full_prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    height=height,
                    width=width,
                    num_inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
                img: Image.Image = result.images[0]
        except Exception as e:
            st.error(f"Generation error for image {idx+1}: {e}")
            continue

        fpath = save_image(img, main_prompt, full_style, full_mood, seed, idx)
        images.append(img)
        filenames.append(fpath)
        generation_logs.append({
            "index": idx + 1,
            "prompt": full_prompt,
            "negative": negative_prompt,
            "steps": inference_steps,
            "guidance": guidance_scale,
            "seed": seed,
            "filename": os.path.basename(fpath)
        })
    
    progress_bar.progress(1.0)
    status_text.text("✅ Generation complete!")
    
    # ------------------------ 4. SHOW IMAGES & LOGS ------------------------
    
    if images:
        st.success(f"🎉 Generated {len(images)} image(s)!")
        
        # Display images in columns
        if len(images) == 1:
            st.image(images[0], caption=f"Generated Art - {filenames[0].split('/')[-1]}", use_column_width=True)
        else:
            cols = st.columns(min(len(images), 3))  # Max 3 columns
            for i, (img, fname) in enumerate(zip(images, filenames)):
                with cols[i % 3]:
                    st.image(img, caption=f"Variation {i+1}")
                    st.caption(f"Seed: {generation_logs[i]['seed']}")
        
        # Generation details
        with st.expander("📊 Generation Details"):
            for log in generation_logs:
                st.write(f"**Image {log['index']}:**")
                st.write(f"- Prompt: {log['prompt']}")
                if log['negative']:
                    st.write(f"- Negative: {log['negative']}")
                st.write(f"- Steps: {log['steps']}, Guidance: {log['guidance']}, Seed: {log['seed']}")
                st.write(f"- Saved as: `{log['filename']}`")
                st.write("---")
        
        st.info(f"💾 All images saved to: `{OUTPUT_DIR}`")
    
elif submitted:
    st.warning("Please enter a prompt to generate images.")

# ------------------------ 5. FOOTER & INFO ------------------------

st.markdown("---")
st.markdown("""
### 🚀 CompI Phase 1.C - Interactive UI

**Features:**
- 🎨 Text-to-image generation with style and mood control
- 🎭 13 art styles and 11 mood options
- ⚙️ Advanced settings (steps, guidance, dimensions)
- 🔄 Multiple variations with unique seeds
- 💾 Automatic saving with descriptive filenames
- 🚫 Negative prompts for better control

**Next Phases:**
- 🎵 Audio input processing
- 😊 Emotion detection
- 🌐 Real-time data integration
- 🔗 Multimodal fusion
""")

st.caption("Built with Streamlit, Diffusers, and PyTorch | CompI - Compositional Intelligence Platform")

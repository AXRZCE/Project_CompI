import os
import sys
import torch
from datetime import datetime
from diffusers import StableDiffusionPipeline
from PIL import Image
import gradio as gr

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# ------------------------ 1. SETUP ------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Running on: {DEVICE.upper()}")

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

print("🔄 Loading AI model...")
pipe = load_pipe()
print("✅ Model loaded successfully!")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------ 2. STYLE AND MOOD OPTIONS ------------------------

STYLES = [
    "digital art", "oil painting", "watercolor", "cyberpunk", "impressionist", "concept art",
    "anime", "photorealistic", "minimalist", "surrealism", "pixel art", "steampunk", "3d render"
]

MOODS = [
    "dreamy atmosphere", "dark and moody", "peaceful", "vibrant and energetic", "melancholic",
    "mysterious", "whimsical", "serene", "uplifting", "dramatic lighting", "retro"
]

# ------------------------ 3. GENERATION FUNCTION ------------------------

def save_image(img: Image.Image, prompt, style, mood, seed, idx):
    prompt_slug = "_".join(prompt.lower().split()[:5])
    style_slug = style.replace(" ", "")[:10]
    mood_slug = mood.replace(" ", "")[:10] if mood else "none"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{prompt_slug[:25]}_{style_slug}_{mood_slug}_{timestamp}_seed{seed}_v{idx+1}.png"
    fpath = os.path.join(OUTPUT_DIR, fname)
    img.save(fpath)
    return fpath

def generate_art(prompt, style, custom_style, mood, custom_mood, negative_prompt, 
                num_variations, steps, guidance, use_seed, seed_value):
    
    if not prompt.strip():
        return None, "❌ Please enter a prompt!"
    
    # Build full prompt
    full_style = custom_style.strip() if custom_style.strip() else style
    full_mood = custom_mood.strip() if custom_mood.strip() else mood
    
    full_prompt = prompt.strip()
    if full_style: full_prompt += f", {full_style}"
    if full_mood: full_prompt += f", {full_mood}"
    
    images = []
    log_messages = []
    
    log_messages.append(f"🎨 Generating {num_variations} image(s)")
    log_messages.append(f"📝 Full prompt: {full_prompt}")
    if negative_prompt:
        log_messages.append(f"🚫 Negative prompt: {negative_prompt}")
    
    for idx in range(num_variations):
        # Set seed
        if use_seed:
            seed = seed_value + idx
        else:
            seed = torch.seed()
        
        generator = torch.manual_seed(seed) if DEVICE == "cpu" else torch.Generator(DEVICE).manual_seed(seed)
        
        try:
            with torch.autocast(DEVICE) if DEVICE == "cuda" else torch.no_grad():
                result = pipe(
                    full_prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    height=512,
                    width=512,
                    num_inference_steps=int(steps),
                    guidance_scale=guidance,
                    generator=generator,
                )
                img = result.images[0]
            
            # Save image
            fpath = save_image(img, prompt, full_style, full_mood, seed, idx)
            images.append(img)
            
            log_messages.append(f"✅ Image {idx+1}: seed {seed}, saved as {os.path.basename(fpath)}")
            
        except Exception as e:
            log_messages.append(f"❌ Error generating image {idx+1}: {str(e)}")
            continue
    
    log_messages.append(f"🎉 Generated {len(images)} image(s) successfully!")
    log_messages.append(f"💾 Images saved to: {OUTPUT_DIR}")
    
    return images, "\n".join(log_messages)

# ------------------------ 4. GRADIO INTERFACE ------------------------

def create_interface():
    with gr.Blocks(title="CompI - AI Art Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🖼️ CompI: Text+Style+Mood to AI Art
        ### Generate beautiful AI art by combining text descriptions with artistic styles and moods
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎨 Create Your Art")
                
                prompt = gr.Textbox(
                    label="Describe your scene or subject",
                    placeholder="A magical forest with glowing trees",
                    lines=2
                )
                
                with gr.Row():
                    style = gr.Dropdown(
                        choices=STYLES,
                        label="Art Style",
                        value="digital art"
                    )
                    mood = gr.Dropdown(
                        choices=MOODS,
                        label="Mood & Atmosphere",
                        value="dreamy atmosphere"
                    )
                
                with gr.Row():
                    custom_style = gr.Textbox(
                        label="Custom Style (optional)",
                        placeholder="Enter your own style..."
                    )
                    custom_mood = gr.Textbox(
                        label="Custom Mood (optional)",
                        placeholder="Enter your own mood..."
                    )
                
                negative_prompt = gr.Textbox(
                    label="Negative Prompt (what to avoid)",
                    placeholder="blurry, low quality, distorted",
                    value="blurry, low quality, distorted"
                )
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    with gr.Row():
                        num_variations = gr.Slider(
                            minimum=1, maximum=4, value=1, step=1,
                            label="Number of Variations"
                        )
                        steps = gr.Slider(
                            minimum=10, maximum=50, value=30, step=5,
                            label="Inference Steps"
                        )
                    
                    with gr.Row():
                        guidance = gr.Slider(
                            minimum=1.0, maximum=20.0, value=7.5, step=0.5,
                            label="Guidance Scale"
                        )
                        use_seed = gr.Checkbox(
                            label="Use Fixed Seed",
                            value=False
                        )
                    
                    seed_value = gr.Number(
                        label="Seed Value",
                        value=42,
                        visible=False
                    )
                
                generate_btn = gr.Button("🎨 Generate Art", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Generated Images")
                
                gallery = gr.Gallery(
                    label="Your AI Art",
                    show_label=False,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height="auto"
                )
                
                logs = gr.Textbox(
                    label="Generation Logs",
                    lines=8,
                    max_lines=15,
                    interactive=False
                )
        
        # Show/hide seed input based on checkbox
        use_seed.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[use_seed],
            outputs=[seed_value]
        )
        
        # Generate button click
        generate_btn.click(
            fn=generate_art,
            inputs=[
                prompt, style, custom_style, mood, custom_mood, negative_prompt,
                num_variations, steps, guidance, use_seed, seed_value
            ],
            outputs=[gallery, logs]
        )
        
        gr.Markdown("""
        ---
        ### 🚀 CompI Phase 1.C - Interactive UI
        
        **Features:** Text-to-image generation with style and mood control, multiple variations, advanced settings
        
        **Next Phases:** Audio input, emotion detection, real-time data integration, multimodal fusion
        
        *Built with Gradio, Diffusers, and PyTorch*
        """)
    
    return demo

# ------------------------ 5. LAUNCH ------------------------

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

# compi_phase1_text2image.py

import os
import sys
import torch
from datetime import datetime
from diffusers import StableDiffusionPipeline
from PIL import Image

# ------------------ 1. SETUP AND CHECKS ------------------

# Check for GPU
if torch.cuda.is_available():
    device = "cuda"
    print("CUDA GPU detected. Running on GPU for best performance.")
else:
    device = "cpu"
    print("No CUDA GPU detected. Running on CPU. Generation will be slow.")

# Set up output directory
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging function
def log(msg):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {msg}")

# ------------------ 2. LOAD MODEL ------------------

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
log(f"Loading model: {MODEL_NAME} (this may take a minute on first run)")

# Optionally, disable the safety checker for pure creative exploration
def dummy_safety_checker(images, **kwargs):
    return images, False

try:
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=dummy_safety_checker,  # Remove for production!
    )
except Exception as e:
    log(f"Error loading model: {e}")
    sys.exit(1)

pipe = pipe.to(device)
pipe.enable_attention_slicing()  # Reduce VRAM use

log("Model loaded successfully.")

# ------------------ 3. PROMPT HANDLING ------------------

if len(sys.argv) > 1:
    prompt = " ".join(sys.argv[1:])
    log(f"Prompt taken from command line: {prompt}")
else:
    prompt = input("Enter your prompt (e.g. 'A magical forest, digital art'): ").strip()
    log(f"Prompt entered: {prompt}")

if not prompt:
    log("No prompt provided. Exiting.")
    sys.exit(0)

# ------------------ 4. GENERATION PARAMETERS ------------------

SEED = torch.seed()  # You can use a fixed seed for reproducibility, e.g. 1234
generator = torch.manual_seed(SEED) if device == "cpu" else torch.Generator(device).manual_seed(torch.seed())

num_inference_steps = 30   # More steps = better quality, slower (default 50)
guidance_scale = 7.5       # Higher = follow prompt more strictly

# Output image size (SDv1.5 default 512x512)
height = 512
width = 512

# ------------------ 5. IMAGE GENERATION ------------------

log(f"Generating image for prompt: {prompt}")
log(f"Params: steps={num_inference_steps}, guidance_scale={guidance_scale}, seed={SEED}")

with torch.autocast(device) if device == "cuda" else torch.no_grad():
    result = pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    image: Image.Image = result.images[0]

# ------------------ 6. SAVE OUTPUT ------------------

# Filename: prompt short, datetime, seed
prompt_slug = "_".join(prompt.lower().split()[:6])
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{prompt_slug[:40]}_{timestamp}_seed{SEED}.png"
filepath = os.path.join(OUTPUT_DIR, filename)
image.save(filepath)
log(f"Image saved to {filepath}")

# Optionally, show image (uncomment next line if running locally)
# image.show()

# Log end
log("Generation complete.")

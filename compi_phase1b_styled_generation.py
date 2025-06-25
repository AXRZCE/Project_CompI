# compi_phase1b_styled_generation.py

import os
import sys
import torch
from datetime import datetime
from diffusers import StableDiffusionPipeline
from PIL import Image

# -------- 1. SETUP --------
if torch.cuda.is_available():
    device = "cuda"
    print("Running on CUDA GPU.")
else:
    device = "cpu"
    print("Running on CPU.")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {msg}")

# -------- 2. LOAD MODEL --------
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
log(f"Loading model: {MODEL_NAME}")

def dummy_safety_checker(images, **kwargs):
    return images, [False] * len(images)

try:
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=dummy_safety_checker,
    )
except Exception as e:
    log(f"Error loading model: {e}")
    sys.exit(1)

pipe = pipe.to(device)
pipe.enable_attention_slicing()

log("Model loaded.")

# -------- 3. STYLE & MOOD PROMPT ENGINEERING --------

# Predefined styles and moods (add more as desired)
STYLES = [
    "digital art",
    "oil painting",
    "watercolor",
    "cyberpunk",
    "impressionist",
    "concept art",
    "anime",
    "photorealistic",
    "minimalist",
    "surrealism",
    "pixel art",
    "steampunk",
    "3d render"
]

MOODS = [
    "dreamy atmosphere",
    "dark and moody",
    "peaceful",
    "vibrant and energetic",
    "melancholic",
    "mysterious",
    "whimsical",
    "serene",
    "uplifting",
    "dramatic lighting",
    "retro"
]

# Input: main prompt
if len(sys.argv) > 1:
    main_prompt = " ".join(sys.argv[1:])
    log(f"Prompt from command line: {main_prompt}")
else:
    main_prompt = input("Enter your main scene/subject (e.g., 'A forest of bioluminescent trees'): ").strip()

if not main_prompt:
    log("No main prompt entered. Exiting.")
    sys.exit(0)

# Style selector
print("\nChoose an art style from the list or enter your own:")
for idx, style in enumerate(STYLES, 1):
    print(f"  {idx}. {style}")
style_choice = input(f"Enter style number [1-{len(STYLES)}] or type your own: ").strip()
if style_choice.isdigit() and 1 <= int(style_choice) <= len(STYLES):
    style = STYLES[int(style_choice)-1]
else:
    style = style_choice if style_choice else STYLES[0]
log(f"Style selected: {style}")

# Mood selector
print("\nChoose a mood from the list or enter your own:")
for idx, mood in enumerate(MOODS, 1):
    print(f"  {idx}. {mood}")
mood_choice = input(f"Enter mood number [1-{len(MOODS)}] or type your own (or leave blank): ").strip()
if mood_choice.isdigit() and 1 <= int(mood_choice) <= len(MOODS):
    mood = MOODS[int(mood_choice)-1]
elif mood_choice:
    mood = mood_choice
else:
    mood = ""
log(f"Mood selected: {mood if mood else '[none]'}")

# Combine all for final prompt
full_prompt = main_prompt
if style: full_prompt += f", {style}"
if mood:  full_prompt += f", {mood}"
log(f"Full prompt: {full_prompt}")

# -------- 4. GENERATION PARAMETERS --------

NUM_VARIATIONS = input("How many variations to generate? (default 1): ").strip()
try:
    NUM_VARIATIONS = max(1, int(NUM_VARIATIONS))
except Exception:
    NUM_VARIATIONS = 1

INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
HEIGHT = 512
WIDTH = 512

# -------- 5. IMAGE GENERATION --------

log(f"Generating {NUM_VARIATIONS} image(s) for prompt: '{full_prompt}'")
images = []

for i in range(NUM_VARIATIONS):
    seed = torch.seed()  # random seed for each variation
    generator = torch.manual_seed(seed) if device == "cpu" else torch.Generator(device).manual_seed(seed)
    with torch.autocast(device) if device == "cuda" else torch.no_grad():
        result = pipe(
            full_prompt,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        )
        img: Image.Image = result.images[0]
        # Compose filename
        prompt_slug = "_".join(main_prompt.lower().split()[:5])
        style_slug = style.replace(" ", "")[:10]
        mood_slug = mood.replace(" ", "")[:10] if mood else "none"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{prompt_slug[:25]}_{style_slug}_{mood_slug}_{timestamp}_seed{seed}_v{i+1}.png"
        fpath = os.path.join(OUTPUT_DIR, fname)
        img.save(fpath)
        log(f"Image saved: {fpath}")
        images.append(fpath)

log(f"All {NUM_VARIATIONS} images generated and saved.")
log("Phase 1.B complete.")

# compi_phase1b_advanced_styling.py
# Advanced style conditioning with negative prompts, quality settings, and enhanced prompt engineering

import os
import sys
import torch
import json
import argparse
from datetime import datetime
from diffusers import StableDiffusionPipeline
from PIL import Image

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# -------- 1. SETUP AND ARGUMENT PARSING --------

def setup_args():
    parser = argparse.ArgumentParser(description="CompI Phase 1.B: Advanced Style Conditioning")
    parser.add_argument("prompt", nargs="*", help="Main scene/subject description")
    parser.add_argument("--style", "-s", help="Art style (or number from list)")
    parser.add_argument("--mood", "-m", help="Mood/atmosphere (or number from list)")
    parser.add_argument("--variations", "-v", type=int, default=1, help="Number of variations")
    parser.add_argument("--quality", "-q", choices=["draft", "standard", "high"], default="standard", help="Quality preset")
    parser.add_argument("--negative", "-n", help="Negative prompt")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--list-styles", action="store_true", help="List available styles and exit")
    parser.add_argument("--list-moods", action="store_true", help="List available moods and exit")
    return parser.parse_args()

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {msg}")

# -------- 2. STYLE AND MOOD DEFINITIONS --------

STYLES = {
    "digital art": {
        "prompt": "digital art, highly detailed",
        "negative": "blurry, pixelated, low resolution"
    },
    "oil painting": {
        "prompt": "oil painting, classical art, brushstrokes, canvas texture",
        "negative": "digital, pixelated, modern"
    },
    "watercolor": {
        "prompt": "watercolor painting, soft colors, flowing paint",
        "negative": "harsh lines, digital, photographic"
    },
    "cyberpunk": {
        "prompt": "cyberpunk style, neon lights, futuristic, sci-fi",
        "negative": "natural, organic, pastoral"
    },
    "impressionist": {
        "prompt": "impressionist painting, soft brushstrokes, light and color",
        "negative": "sharp details, photorealistic, digital"
    },
    "concept art": {
        "prompt": "concept art, professional illustration, detailed",
        "negative": "amateur, sketch, unfinished"
    },
    "anime": {
        "prompt": "anime style, manga, Japanese animation",
        "negative": "realistic, western cartoon, photographic"
    },
    "photorealistic": {
        "prompt": "photorealistic, high detail, professional photography",
        "negative": "cartoon, painting, stylized"
    },
    "minimalist": {
        "prompt": "minimalist art, clean lines, simple composition",
        "negative": "cluttered, complex, detailed"
    },
    "surrealism": {
        "prompt": "surrealist art, dreamlike, impossible, Salvador Dali style",
        "negative": "realistic, logical, mundane"
    },
    "pixel art": {
        "prompt": "pixel art, 8-bit style, retro gaming",
        "negative": "smooth, high resolution, photorealistic"
    },
    "steampunk": {
        "prompt": "steampunk style, Victorian era, brass and copper, gears",
        "negative": "modern, digital, futuristic"
    },
    "3d render": {
        "prompt": "3D render, CGI, computer graphics, ray tracing",
        "negative": "2D, flat, hand-drawn"
    }
}

MOODS = {
    "dreamy": {
        "prompt": "dreamy atmosphere, soft lighting, ethereal",
        "negative": "harsh, stark, realistic"
    },
    "dark": {
        "prompt": "dark and moody, dramatic shadows, mysterious",
        "negative": "bright, cheerful, light"
    },
    "peaceful": {
        "prompt": "peaceful, serene, calm, tranquil",
        "negative": "chaotic, violent, disturbing"
    },
    "vibrant": {
        "prompt": "vibrant and energetic, bright colors, dynamic",
        "negative": "dull, muted, lifeless"
    },
    "melancholic": {
        "prompt": "melancholic, sad, nostalgic, wistful",
        "negative": "happy, joyful, upbeat"
    },
    "mysterious": {
        "prompt": "mysterious, enigmatic, hidden secrets",
        "negative": "obvious, clear, straightforward"
    },
    "whimsical": {
        "prompt": "whimsical, playful, fantastical, magical",
        "negative": "serious, realistic, mundane"
    },
    "dramatic": {
        "prompt": "dramatic lighting, high contrast, cinematic",
        "negative": "flat lighting, low contrast, amateur"
    },
    "retro": {
        "prompt": "retro style, vintage, nostalgic, classic",
        "negative": "modern, contemporary, futuristic"
    }
}

QUALITY_PRESETS = {
    "draft": {"steps": 20, "guidance": 6.0, "size": (512, 512)},
    "standard": {"steps": 30, "guidance": 7.5, "size": (512, 512)},
    "high": {"steps": 50, "guidance": 8.5, "size": (768, 768)}
}

# -------- 3. MODEL LOADING --------

def load_model():
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
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        log("Model loaded successfully")
        return pipe
    except Exception as e:
        log(f"Error loading model: {e}")
        sys.exit(1)

# -------- 4. INTERACTIVE FUNCTIONS --------

def list_options(options_dict, title):
    print(f"\n{title}:")
    for idx, (key, value) in enumerate(options_dict.items(), 1):
        prompt_preview = value["prompt"][:50] + "..." if len(value["prompt"]) > 50 else value["prompt"]
        print(f"  {idx:2d}. {key:15s} - {prompt_preview}")

def get_user_choice(options_dict, prompt_text, allow_custom=True):
    choice = input(f"{prompt_text}: ").strip()
    
    if choice.isdigit():
        idx = int(choice) - 1
        keys = list(options_dict.keys())
        if 0 <= idx < len(keys):
            return keys[idx]
    
    if choice in options_dict:
        return choice
    
    if allow_custom and choice:
        return choice
    
    return None

def interactive_mode(pipe):
    log("Starting interactive style conditioning mode")
    
    # Get main prompt
    main_prompt = input("\nEnter your main scene/subject: ").strip()
    if not main_prompt:
        log("No prompt provided")
        return
    
    # Show and select style
    list_options(STYLES, "Available Styles")
    style_key = get_user_choice(STYLES, "Choose style (number/name/custom)")
    
    # Show and select mood
    list_options(MOODS, "Available Moods")
    mood_key = get_user_choice(MOODS, "Choose mood (number/name/custom/blank)", allow_custom=True)
    
    # Get additional parameters
    variations = input("Number of variations (default 1): ").strip()
    variations = int(variations) if variations.isdigit() else 1
    
    quality = input("Quality [draft/standard/high] (default standard): ").strip()
    quality = quality if quality in QUALITY_PRESETS else "standard"
    
    negative = input("Negative prompt (optional): ").strip()
    
    # Generate images
    generate_styled_images(pipe, main_prompt, style_key, mood_key, variations, quality, negative)

# -------- 5. GENERATION FUNCTION --------

def generate_styled_images(pipe, main_prompt, style_key, mood_key, variations, quality, custom_negative=""):
    # Build the full prompt
    full_prompt = main_prompt
    style_negative = ""
    mood_negative = ""
    
    if style_key and style_key in STYLES:
        full_prompt += f", {STYLES[style_key]['prompt']}"
        style_negative = STYLES[style_key]['negative']
    elif style_key:
        full_prompt += f", {style_key}"
    
    if mood_key and mood_key in MOODS:
        full_prompt += f", {MOODS[mood_key]['prompt']}"
        mood_negative = MOODS[mood_key]['negative']
    elif mood_key:
        full_prompt += f", {mood_key}"
    
    # Build negative prompt
    negative_parts = [part for part in [style_negative, mood_negative, custom_negative] if part]
    full_negative = ", ".join(negative_parts) if negative_parts else None
    
    # Get quality settings
    quality_settings = QUALITY_PRESETS[quality]
    
    log(f"Full prompt: {full_prompt}")
    log(f"Negative prompt: {full_negative or '[none]'}")
    log(f"Quality: {quality} ({quality_settings['steps']} steps)")
    log(f"Generating {variations} variation(s)")
    
    # Generate images
    for i in range(variations):
        seed = torch.seed()
        generator = torch.manual_seed(seed) if device == "cpu" else torch.Generator(device).manual_seed(seed)
        
        with torch.autocast(device) if device == "cuda" else torch.no_grad():
            result = pipe(
                full_prompt,
                negative_prompt=full_negative,
                height=quality_settings["size"][1],
                width=quality_settings["size"][0],
                num_inference_steps=quality_settings["steps"],
                guidance_scale=quality_settings["guidance"],
                generator=generator,
            )
            
            img = result.images[0]
            
            # Create filename
            prompt_slug = "_".join(main_prompt.lower().split()[:4])
            style_slug = (style_key or "nostyle").replace(" ", "")[:10]
            mood_slug = (mood_key or "nomood").replace(" ", "")[:10]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            filename = f"{prompt_slug[:20]}_{style_slug}_{mood_slug}_{quality}_{timestamp}_seed{seed}_v{i+1}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            img.save(filepath)
            
            # Save metadata
            metadata = {
                "main_prompt": main_prompt,
                "style": style_key,
                "mood": mood_key,
                "full_prompt": full_prompt,
                "negative_prompt": full_negative,
                "quality": quality,
                "seed": seed,
                "variation": i + 1,
                "timestamp": datetime.now().isoformat(),
                "settings": quality_settings
            }
            
            metadata_file = filepath.replace('.png', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            log(f"Generated variation {i+1}: {filepath}")
    
    log(f"Phase 1.B complete - {variations} styled images generated")

# -------- 6. MAIN FUNCTION --------

def main():
    args = setup_args()
    
    # Handle list commands
    if args.list_styles:
        list_options(STYLES, "Available Styles")
        return
    
    if args.list_moods:
        list_options(MOODS, "Available Moods")
        return
    
    # Load model
    pipe = load_model()
    
    # Interactive mode
    if args.interactive:
        interactive_mode(pipe)
        return
    
    # Command line mode
    main_prompt = " ".join(args.prompt) if args.prompt else input("Enter main prompt: ").strip()
    if not main_prompt:
        log("No prompt provided")
        return
    
    generate_styled_images(
        pipe, main_prompt, args.style, args.mood, 
        args.variations, args.quality, args.negative or ""
    )

if __name__ == "__main__":
    main()

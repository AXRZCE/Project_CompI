# compi_phase1_advanced.py
# Enhanced text-to-image generation with batch processing, negative prompts, and style controls

import os
import sys
import torch
import argparse
from datetime import datetime
from diffusers import StableDiffusionPipeline
from PIL import Image
import json

# ------------------ 1. SETUP AND ARGUMENT PARSING ------------------

def setup_args():
    parser = argparse.ArgumentParser(description="CompI Phase 1: Advanced Text-to-Image Generation")
    parser.add_argument("prompt", nargs="*", help="Text prompt for image generation")
    parser.add_argument("--negative", "-n", default="", help="Negative prompt (what to avoid)")
    parser.add_argument("--steps", "-s", type=int, default=30, help="Number of inference steps (default: 30)")
    parser.add_argument("--guidance", "-g", type=float, default=7.5, help="Guidance scale (default: 7.5)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--batch", "-b", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--width", "-w", type=int, default=512, help="Image width (default: 512)")
    parser.add_argument("--height", type=int, default=512, help="Image height (default: 512)")
    parser.add_argument("--model", "-m", default="runwayml/stable-diffusion-v1-5", help="Model to use")
    parser.add_argument("--output", "-o", default="outputs", help="Output directory")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    return parser.parse_args()

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Logging function
def log(msg):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {msg}")

# ------------------ 2. MODEL LOADING ------------------

def load_model(model_name):
    log(f"Loading model: {model_name}")
    
    def dummy_safety_checker(images, **kwargs):
        return images, [False] * len(images)
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=dummy_safety_checker,
        )
        pipe = pipe.to(device)
        
        # Memory optimizations
        pipe.enable_attention_slicing()
        # Note: enable_memory_efficient_attention() is deprecated in newer versions
        
        log("Model loaded successfully")
        return pipe
    except Exception as e:
        log(f"Error loading model: {e}")
        sys.exit(1)

# ------------------ 3. GENERATION FUNCTION ------------------

def generate_image(pipe, prompt, negative_prompt="", **kwargs):
    """Generate a single image with given parameters"""
    
    # Set up generator
    seed = kwargs.get('seed', torch.seed())
    if device == "cuda":
        generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = torch.manual_seed(seed)
    
    # Generation parameters
    params = {
        'prompt': prompt,
        'negative_prompt': negative_prompt if negative_prompt else None,
        'height': kwargs.get('height', 512),
        'width': kwargs.get('width', 512),
        'num_inference_steps': kwargs.get('steps', 30),
        'guidance_scale': kwargs.get('guidance', 7.5),
        'generator': generator,
    }
    
    log(f"Generating: '{prompt[:50]}...' (seed: {seed})")
    
    with torch.autocast(device) if device == "cuda" else torch.no_grad():
        result = pipe(**params)
        return result.images[0], seed

# ------------------ 4. SAVE FUNCTION ------------------

def save_image(image, prompt, seed, output_dir, metadata=None):
    """Save image with descriptive filename and metadata"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename
    prompt_slug = "_".join(prompt.lower().split()[:6])
    prompt_slug = "".join(c for c in prompt_slug if c.isalnum() or c in "_-")[:40]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prompt_slug}_{timestamp}_seed{seed}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Save image
    image.save(filepath)
    
    # Save metadata
    if metadata:
        metadata_file = filepath.replace('.png', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    log(f"Saved: {filepath}")
    return filepath

# ------------------ 5. INTERACTIVE MODE ------------------

def interactive_mode(pipe, output_dir):
    """Interactive prompt mode for experimentation"""
    log("Entering interactive mode. Type 'quit' to exit.")
    
    while True:
        try:
            prompt = input("\n🎨 Enter prompt: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt:
                continue
                
            negative = input("❌ Negative prompt (optional): ").strip()
            
            # Quick settings
            print("⚙️  Quick settings (press Enter for defaults):")
            steps = input(f"   Steps (30): ").strip()
            steps = int(steps) if steps else 30
            
            guidance = input(f"   Guidance (7.5): ").strip()
            guidance = float(guidance) if guidance else 7.5
            
            # Generate
            image, seed = generate_image(
                pipe, prompt, negative,
                steps=steps, guidance=guidance
            )
            
            # Save with metadata
            metadata = {
                'prompt': prompt,
                'negative_prompt': negative,
                'steps': steps,
                'guidance_scale': guidance,
                'seed': seed,
                'timestamp': datetime.now().isoformat()
            }
            
            save_image(image, prompt, seed, output_dir, metadata)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            log(f"Error: {e}")

# ------------------ 6. MAIN FUNCTION ------------------

def main():
    args = setup_args()
    
    # Load model
    pipe = load_model(args.model)
    
    # Interactive mode
    if args.interactive:
        interactive_mode(pipe, args.output)
        return
    
    # Get prompt
    if args.prompt:
        prompt = " ".join(args.prompt)
    else:
        prompt = input("Enter your prompt: ").strip()
    
    if not prompt:
        log("No prompt provided. Use --interactive for interactive mode.")
        return
    
    # Generate batch
    log(f"Generating {args.batch} image(s)")
    
    for i in range(args.batch):
        try:
            # Use provided seed or generate random one
            seed = args.seed if args.seed else torch.seed()
            
            image, actual_seed = generate_image(
                pipe, prompt, args.negative,
                seed=seed, steps=args.steps, guidance=args.guidance,
                width=args.width, height=args.height
            )
            
            # Prepare metadata
            metadata = {
                'prompt': prompt,
                'negative_prompt': args.negative,
                'steps': args.steps,
                'guidance_scale': args.guidance,
                'seed': actual_seed,
                'width': args.width,
                'height': args.height,
                'model': args.model,
                'batch_index': i + 1,
                'timestamp': datetime.now().isoformat()
            }
            
            save_image(image, prompt, actual_seed, args.output, metadata)
            
        except Exception as e:
            log(f"Error generating image {i+1}: {e}")
    
    log("Generation complete!")

if __name__ == "__main__":
    main()

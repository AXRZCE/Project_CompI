#!/usr/bin/env python3
"""
CompI Phase 1.E: Personal Style Generation with LoRA

Generate images using your trained LoRA personal style weights.

Usage:
    python src/generators/compi_phase1e_style_generation.py --lora-path lora_models/my_style/checkpoint-1000
    python src/generators/compi_phase1e_style_generation.py --help
"""

import os
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel

# -------- 1. CONFIGURATION --------

DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 7.5
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
OUTPUT_DIR = "outputs"

# -------- 2. UTILITY FUNCTIONS --------

def setup_args():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(
        description="CompI Phase 1.E: Personal Style Generation with LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with trained LoRA style
  python %(prog)s --lora-path lora_models/my_style/checkpoint-1000 "a cat in my_style"
  
  # Interactive mode
  python %(prog)s --lora-path lora_models/my_style/checkpoint-1000 --interactive
  
  # Multiple variations
  python %(prog)s --lora-path lora_models/my_style/checkpoint-1000 "landscape" --variations 4
        """
    )
    
    parser.add_argument("prompt", nargs="*", help="Text prompt for generation")
    
    parser.add_argument("--lora-path", required=True,
                       help="Path to trained LoRA checkpoint directory")
    
    parser.add_argument("--model-name", default=DEFAULT_MODEL,
                       help=f"Base Stable Diffusion model (default: {DEFAULT_MODEL})")
    
    parser.add_argument("--variations", "-v", type=int, default=1,
                       help="Number of variations to generate")
    
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                       help=f"Number of inference steps (default: {DEFAULT_STEPS})")
    
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE,
                       help=f"Guidance scale (default: {DEFAULT_GUIDANCE})")
    
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                       help=f"Image width (default: {DEFAULT_WIDTH})")
    
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT,
                       help=f"Image height (default: {DEFAULT_HEIGHT})")
    
    parser.add_argument("--seed", type=int,
                       help="Random seed for reproducible generation")
    
    parser.add_argument("--negative", "-n", default="",
                       help="Negative prompt")
    
    parser.add_argument("--lora-scale", type=float, default=1.0,
                       help="LoRA scale factor (0.0-2.0, default: 1.0)")
    
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Interactive mode")
    
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                       help=f"Output directory (default: {OUTPUT_DIR})")
    
    parser.add_argument("--list-styles", action="store_true",
                       help="List available LoRA styles")
    
    return parser.parse_args()

def load_lora_info(lora_path: str) -> dict:
    """Load LoRA training information."""
    lora_dir = Path(lora_path)
    
    # Try to find training info
    info_files = [
        lora_dir / "training_info.json",
        lora_dir.parent / "training_info.json"
    ]
    
    for info_file in info_files:
        if info_file.exists():
            with open(info_file) as f:
                return json.load(f)
    
    # Fallback info
    return {
        'style_name': lora_dir.parent.name,
        'model_name': DEFAULT_MODEL,
        'lora_rank': 4,
        'lora_alpha': 32
    }

def load_pipeline_with_lora(model_name: str, lora_path: str, device: str):
    """Load Stable Diffusion pipeline with LoRA weights."""
    print(f"🔄 Loading base model: {model_name}")
    
    # Load base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Use DPM solver for faster inference
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    print(f"🎨 Loading LoRA weights from: {lora_path}")
    
    # Load LoRA weights
    lora_dir = Path(lora_path)
    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")
    
    # Apply LoRA to UNet
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    
    # Move to device
    pipe = pipe.to(device)
    
    # Enable memory efficient attention if available
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    
    return pipe

def generate_with_style(
    pipe, 
    prompt: str, 
    negative_prompt: str = "",
    num_inference_steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    seed: Optional[int] = None,
    lora_scale: float = 1.0
):
    """Generate image with LoRA style."""
    
    # Set LoRA scale
    if hasattr(pipe.unet, 'set_adapter_scale'):
        pipe.unet.set_adapter_scale(lora_scale)
    
    # Setup generator
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    else:
        generator = None
        seed = torch.seed()
    
    # Generate image
    with torch.autocast(pipe.device.type):
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        )
    
    return result.images[0], seed

def save_generated_image(
    image: Image.Image, 
    prompt: str, 
    style_name: str,
    seed: int,
    variation: int,
    output_dir: str,
    metadata: dict = None
):
    """Save generated image with metadata."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    prompt_slug = "_".join(prompt.lower().split()[:5])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prompt_slug[:25]}_lora_{style_name}_{timestamp}_seed{seed}_v{variation}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Save image
    image.save(filepath)
    
    # Save metadata if provided
    if metadata:
        metadata_file = filepath.replace('.png', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return filepath

def list_available_styles():
    """List available LoRA styles."""
    lora_dir = Path("lora_models")
    
    if not lora_dir.exists():
        print("❌ No LoRA models directory found")
        return
    
    print("🎨 Available LoRA Styles:")
    print("=" * 40)
    
    styles_found = False
    for style_dir in lora_dir.iterdir():
        if style_dir.is_dir():
            # Look for checkpoints
            checkpoints = list(style_dir.glob("checkpoint-*"))
            if checkpoints:
                styles_found = True
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
                
                # Load info if available
                info_file = style_dir / "training_info.json"
                if info_file.exists():
                    with open(info_file) as f:
                        info = json.load(f)
                    print(f"📁 {style_dir.name}")
                    print(f"   Latest: {latest_checkpoint.name}")
                    print(f"   Steps: {info.get('total_steps', 'unknown')}")
                    print(f"   Model: {info.get('model_name', 'unknown')}")
                else:
                    print(f"📁 {style_dir.name}")
                    print(f"   Latest: {latest_checkpoint.name}")
                print()
    
    if not styles_found:
        print("❌ No trained LoRA styles found")
        print("💡 Train a style first using: python src/generators/compi_phase1e_lora_training.py")

def interactive_generation(pipe, lora_info: dict, args):
    """Interactive generation mode."""
    style_name = lora_info.get('style_name', 'custom')
    
    print(f"🎨 Interactive LoRA Style Generation - {style_name}")
    print("=" * 50)
    print("💡 Tips:")
    print(f"   - Include '{style_name}' or trigger words in your prompts")
    print(f"   - Adjust LoRA scale (0.0-2.0) to control style strength")
    print("   - Type 'quit' to exit")
    print()
    
    while True:
        try:
            # Get prompt
            prompt = input("Enter prompt: ").strip()
            if not prompt or prompt.lower() == 'quit':
                break
            
            # Get optional parameters
            variations = input(f"Variations (default: 1): ").strip()
            variations = int(variations) if variations.isdigit() else 1
            
            lora_scale = input(f"LoRA scale (default: {args.lora_scale}): ").strip()
            lora_scale = float(lora_scale) if lora_scale else args.lora_scale
            
            # Generate images
            print(f"🎨 Generating {variations} variation(s)...")
            
            for i in range(variations):
                image, seed = generate_with_style(
                    pipe, prompt, args.negative,
                    args.steps, args.guidance,
                    args.width, args.height,
                    args.seed, lora_scale
                )
                
                # Save image
                filepath = save_generated_image(
                    image, prompt, style_name, seed, i + 1, args.output_dir,
                    {
                        'prompt': prompt,
                        'negative_prompt': args.negative,
                        'style_name': style_name,
                        'lora_scale': lora_scale,
                        'seed': seed,
                        'steps': args.steps,
                        'guidance_scale': args.guidance,
                        'timestamp': datetime.now().isoformat()
                    }
                )
                
                print(f"✅ Saved: {filepath}")
            
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print()

def main():
    """Main function."""
    args = setup_args()
    
    # List styles if requested
    if args.list_styles:
        list_available_styles()
        return 0
    
    # Check LoRA path
    if not os.path.exists(args.lora_path):
        print(f"❌ LoRA path not found: {args.lora_path}")
        return 1
    
    # Load LoRA info
    lora_info = load_lora_info(args.lora_path)
    style_name = lora_info.get('style_name', 'custom')
    
    print(f"🎨 CompI Phase 1.E: Personal Style Generation")
    print(f"Style: {style_name}")
    print("=" * 50)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Load pipeline
    try:
        pipe = load_pipeline_with_lora(args.model_name, args.lora_path, device)
        print("✅ Pipeline loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load pipeline: {e}")
        return 1
    
    # Interactive mode
    if args.interactive:
        interactive_generation(pipe, lora_info, args)
        return 0
    
    # Command line mode
    prompt = " ".join(args.prompt) if args.prompt else input("Enter prompt: ").strip()
    if not prompt:
        print("❌ No prompt provided")
        return 1
    
    print(f"🎨 Generating {args.variations} variation(s) for: {prompt}")
    
    # Generate images
    for i in range(args.variations):
        try:
            image, seed = generate_with_style(
                pipe, prompt, args.negative,
                args.steps, args.guidance,
                args.width, args.height,
                args.seed, args.lora_scale
            )
            
            # Save image
            filepath = save_generated_image(
                image, prompt, style_name, seed, i + 1, args.output_dir,
                {
                    'prompt': prompt,
                    'negative_prompt': args.negative,
                    'style_name': style_name,
                    'lora_scale': args.lora_scale,
                    'seed': seed,
                    'steps': args.steps,
                    'guidance_scale': args.guidance,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            print(f"✅ Generated variation {i + 1}: {filepath}")
            
        except Exception as e:
            print(f"❌ Error generating variation {i + 1}: {e}")
    
    print("🎉 Generation complete!")
    return 0

if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
CompI Phase 1.E: LoRA Fine-tuning for Personal Style

This script implements LoRA (Low-Rank Adaptation) fine-tuning for Stable Diffusion
to learn your personal artistic style.

Usage:
    python src/generators/compi_phase1e_lora_training.py --dataset-dir datasets/my_style
    python src/generators/compi_phase1e_lora_training.py --help
"""

import os
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

# Diffusers and transformers
from diffusers import (
    StableDiffusionPipeline, 
    UNet2DConditionModel, 
    DDPMScheduler,
    AutoencoderKL
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# -------- 1. CONFIGURATION --------

DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_RESOLUTION = 512
DEFAULT_BATCH_SIZE = 1
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_EPOCHS = 100
DEFAULT_LORA_RANK = 4
DEFAULT_LORA_ALPHA = 32

# -------- 2. DATASET CLASS --------

class StyleDataset(Dataset):
    """Dataset class for LoRA fine-tuning."""
    
    def __init__(self, dataset_dir: str, split: str = "train", resolution: int = 512):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.resolution = resolution
        
        # Load images and captions
        self.images_dir = self.dataset_dir / split
        self.captions_file = self.dataset_dir / f"{split}_captions.txt"
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        if not self.captions_file.exists():
            raise FileNotFoundError(f"Captions file not found: {self.captions_file}")
        
        # Load captions
        self.image_captions = {}
        with open(self.captions_file, 'r') as f:
            for line in f:
                if ':' in line:
                    filename, caption = line.strip().split(':', 1)
                    self.image_captions[filename.strip()] = caption.strip()
        
        # Get list of images
        self.image_files = [f for f in os.listdir(self.images_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Filter to only images with captions
        self.image_files = [f for f in self.image_files if f in self.image_captions]
        
        print(f"Loaded {len(self.image_files)} images for {split} split")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image_path = self.images_dir / filename
        caption = self.image_captions[filename]
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize to [-1, 1]
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return {
            'pixel_values': image,
            'caption': caption,
            'filename': filename
        }

# -------- 3. TRAINING FUNCTIONS --------

def setup_args():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(
        description="CompI Phase 1.E: LoRA Fine-tuning for Personal Style",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--dataset-dir", required=True,
                       help="Directory containing prepared dataset")
    
    parser.add_argument("--output-dir",
                       help="Output directory for LoRA weights (default: lora_models/{style_name})")
    
    parser.add_argument("--model-name", default=DEFAULT_MODEL,
                       help=f"Base Stable Diffusion model (default: {DEFAULT_MODEL})")
    
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                       help=f"Training resolution (default: {DEFAULT_RESOLUTION})")
    
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                       help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})")
    
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
                       help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})")
    
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                       help=f"Number of training epochs (default: {DEFAULT_EPOCHS})")
    
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK,
                       help=f"LoRA rank (default: {DEFAULT_LORA_RANK})")
    
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA,
                       help=f"LoRA alpha (default: {DEFAULT_LORA_ALPHA})")
    
    parser.add_argument("--save-steps", type=int, default=100,
                       help="Save checkpoint every N steps")
    
    parser.add_argument("--validation-steps", type=int, default=50,
                       help="Run validation every N steps")
    
    parser.add_argument("--mixed-precision", action="store_true",
                       help="Use mixed precision training")
    
    parser.add_argument("--gradient-checkpointing", action="store_true",
                       help="Use gradient checkpointing to save memory")
    
    return parser.parse_args()

def load_models(model_name: str, device: str):
    """Load Stable Diffusion components."""
    print(f"Loading models from {model_name}...")
    
    # Load tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
    
    # Load noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    
    # Move to device
    text_encoder.to(device)
    vae.to(device)
    unet.to(device)
    
    # Set to eval mode (we only train LoRA adapters)
    text_encoder.eval()
    vae.eval()
    unet.train()  # UNet needs to be in train mode for LoRA
    
    return tokenizer, text_encoder, vae, unet, noise_scheduler

def setup_lora(unet: UNet2DConditionModel, lora_rank: int, lora_alpha: int):
    """Setup LoRA adapters for UNet."""
    print(f"Setting up LoRA with rank={lora_rank}, alpha={lora_alpha}")
    
    # Define LoRA config
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "to_k", "to_q", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2"
        ],
        lora_dropout=0.1,
    )
    
    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return unet

def encode_text(tokenizer, text_encoder, captions: List[str], device: str):
    """Encode text captions."""
    inputs = tokenizer(
        captions,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        text_embeddings = text_encoder(inputs.input_ids.to(device))[0]
    
    return text_embeddings

def training_step(batch, unet, vae, text_encoder, tokenizer, noise_scheduler, device):
    """Single training step."""
    pixel_values = batch['pixel_values'].to(device)
    captions = batch['caption']
    
    # Encode images to latent space
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    
    # Sample noise
    noise = torch.randn_like(latents)
    batch_size = latents.shape[0]
    
    # Sample random timesteps
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, 
        (batch_size,), device=device
    ).long()
    
    # Add noise to latents
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    
    # Encode text
    text_embeddings = encode_text(tokenizer, text_encoder, captions, device)
    
    # Predict noise
    noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
    
    # Calculate loss
    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
    
    return loss

def validate_model(val_dataloader, unet, vae, text_encoder, tokenizer, noise_scheduler, device):
    """Validation step."""
    unet.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            loss = training_step(batch, unet, vae, text_encoder, tokenizer, noise_scheduler, device)
            total_loss += loss.item()
            num_batches += 1
    
    unet.train()
    return total_loss / num_batches if num_batches > 0 else 0

def save_lora_weights(unet, output_dir: Path, step: int):
    """Save LoRA weights."""
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LoRA weights
    unet.save_pretrained(checkpoint_dir)
    
    print(f"💾 Saved checkpoint to: {checkpoint_dir}")
    return checkpoint_dir

# -------- 4. MAIN TRAINING FUNCTION --------

def train_lora(args):
    """Main training function."""
    print(f"🎨 CompI Phase 1.E: Starting LoRA Training")
    print("=" * 50)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Load dataset info
    dataset_dir = Path(args.dataset_dir)
    info_file = dataset_dir / "dataset_info.json"
    
    if info_file.exists():
        with open(info_file) as f:
            dataset_info = json.load(f)
        style_name = dataset_info.get('style_name', 'custom_style')
        print(f"🎯 Training style: {style_name}")
    else:
        style_name = dataset_dir.name
        print(f"⚠️  No dataset info found, using directory name: {style_name}")
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("lora_models") / style_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Load datasets
    print(f"📊 Loading datasets...")
    train_dataset = StyleDataset(args.dataset_dir, "train", args.resolution)
    
    try:
        val_dataset = StyleDataset(args.dataset_dir, "validation", args.resolution)
        has_validation = True
    except FileNotFoundError:
        print("⚠️  No validation set found, using train set for validation")
        val_dataset = train_dataset
        has_validation = False
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Load models
    tokenizer, text_encoder, vae, unet, noise_scheduler = load_models(args.model_name, device)
    
    # Setup LoRA
    unet = setup_lora(unet, args.lora_rank, args.lora_alpha)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-08
    )
    
    # Calculate total steps
    total_steps = len(train_dataloader) * args.epochs
    print(f"📈 Total training steps: {total_steps}")
    
    # Training loop
    print(f"\n🚀 Starting training...")
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n📅 Epoch {epoch + 1}/{args.epochs}")
        
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Training")
        
        for batch in progress_bar:
            # Training step
            loss = training_step(batch, unet, vae, text_encoder, tokenizer, noise_scheduler, device)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{epoch_loss / (progress_bar.n + 1):.4f}"
            })
            
            # Validation
            if global_step % args.validation_steps == 0:
                val_loss = validate_model(val_dataloader, unet, vae, text_encoder, tokenizer, noise_scheduler, device)
                print(f"\n📊 Step {global_step}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_lora_weights(unet, output_dir, global_step)
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                save_lora_weights(unet, output_dir, global_step)
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"📊 Epoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")
    
    # Save final model
    final_checkpoint = save_lora_weights(unet, output_dir, global_step)
    
    # Save training info
    training_info = {
        'style_name': style_name,
        'model_name': args.model_name,
        'total_steps': global_step,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'final_checkpoint': str(final_checkpoint),
        'best_val_loss': best_val_loss
    }
    
    with open(output_dir / "training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\n🎉 Training complete!")
    print(f"📁 LoRA weights saved to: {output_dir}")
    print(f"💡 Next steps:")
    print(f"   1. Test your style: python src/generators/compi_phase1e_style_generation.py --lora-path {final_checkpoint}")
    print(f"   2. Integrate with UI: Use the style in your Streamlit interface")

def main():
    """Main function."""
    args = setup_args()
    
    try:
        train_lora(args)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

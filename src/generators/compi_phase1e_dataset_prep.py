#!/usr/bin/env python3
"""
CompI Phase 1.E: Dataset Preparation for LoRA Fine-tuning

This tool helps prepare your personal style dataset for LoRA training:
- Organize and validate style images
- Generate appropriate captions
- Resize and format images for training
- Create training/validation splits

Usage:
    python src/generators/compi_phase1e_dataset_prep.py --help
    python src/generators/compi_phase1e_dataset_prep.py --input-dir my_style_images --style-name "my_art_style"
"""

import os
import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import random

from PIL import Image, ImageOps
import pandas as pd

# -------- 1. CONFIGURATION --------

DEFAULT_IMAGE_SIZE = 512
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
MIN_IMAGES_RECOMMENDED = 10
TRAIN_SPLIT_RATIO = 0.8

# -------- 2. UTILITY FUNCTIONS --------

def setup_args():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(
        description="CompI Phase 1.E: Dataset Preparation for LoRA Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare dataset from a folder of images
  python %(prog)s --input-dir my_artwork --style-name "impressionist_style"
  
  # Custom output directory and image size
  python %(prog)s --input-dir paintings --style-name "oil_painting" --output-dir datasets/oil_style --size 768
  
  # Generate captions with custom trigger word
  python %(prog)s --input-dir sketches --style-name "pencil_sketch" --trigger-word "sketch_style"
        """
    )
    
    parser.add_argument("--input-dir", required=True,
                       help="Directory containing your style images")
    
    parser.add_argument("--style-name", required=True,
                       help="Name for your style (used in file naming and captions)")
    
    parser.add_argument("--output-dir", 
                       help="Output directory for prepared dataset (default: datasets/{style_name})")
    
    parser.add_argument("--trigger-word",
                       help="Trigger word for style (default: style_name)")
    
    parser.add_argument("--size", type=int, default=DEFAULT_IMAGE_SIZE,
                       help=f"Target image size in pixels (default: {DEFAULT_IMAGE_SIZE})")
    
    parser.add_argument("--caption-template", 
                       default="a painting in {trigger_word} style",
                       help="Template for generating captions")
    
    parser.add_argument("--train-split", type=float, default=TRAIN_SPLIT_RATIO,
                       help=f"Ratio for train/validation split (default: {TRAIN_SPLIT_RATIO})")
    
    parser.add_argument("--copy-images", action="store_true",
                       help="Copy images instead of creating symlinks")
    
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate input directory without processing")
    
    return parser.parse_args()

def validate_image_directory(input_dir: str) -> Tuple[List[str], List[str]]:
    """Validate input directory and return valid/invalid image files."""
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    all_files = os.listdir(input_dir)
    valid_images = []
    invalid_files = []
    
    for filename in all_files:
        filepath = os.path.join(input_dir, filename)
        
        # Check if it's a file
        if not os.path.isfile(filepath):
            continue
            
        # Check extension
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            invalid_files.append(f"{filename} (unsupported format: {ext})")
            continue
        
        # Try to open image
        try:
            with Image.open(filepath) as img:
                # Basic validation
                if img.size[0] < 64 or img.size[1] < 64:
                    invalid_files.append(f"{filename} (too small: {img.size})")
                    continue
                    
                valid_images.append(filename)
        except Exception as e:
            invalid_files.append(f"{filename} (corrupt: {str(e)})")
    
    return valid_images, invalid_files

def process_image(input_path: str, output_path: str, target_size: int) -> Dict:
    """Process a single image for training."""
    with Image.open(input_path) as img:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get original dimensions
        original_size = img.size
        
        # Resize maintaining aspect ratio, then center crop
        img = ImageOps.fit(img, (target_size, target_size), Image.Resampling.LANCZOS)
        
        # Save processed image
        img.save(output_path, 'PNG', quality=95)
        
        return {
            'original_size': original_size,
            'processed_size': img.size,
            'format': 'PNG'
        }

def generate_captions(image_files: List[str], caption_template: str, trigger_word: str) -> Dict[str, str]:
    """Generate captions for training images."""
    captions = {}
    
    for filename in image_files:
        # Basic caption using template
        caption = caption_template.format(trigger_word=trigger_word)
        
        # You could add more sophisticated caption generation here
        # For example, using BLIP or other image captioning models
        
        captions[filename] = caption
    
    return captions

def create_dataset_structure(output_dir: str, style_name: str):
    """Create the dataset directory structure."""
    dataset_dir = Path(output_dir)
    
    # Create main directories
    dirs_to_create = [
        dataset_dir,
        dataset_dir / "images",
        dataset_dir / "train",
        dataset_dir / "validation"
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dataset_dir

def split_dataset(image_files: List[str], train_ratio: float) -> Tuple[List[str], List[str]]:
    """Split images into train and validation sets."""
    random.shuffle(image_files)
    
    train_count = int(len(image_files) * train_ratio)
    train_files = image_files[:train_count]
    val_files = image_files[train_count:]
    
    return train_files, val_files

def save_metadata(dataset_dir: Path, metadata: Dict):
    """Save dataset metadata."""
    metadata_file = dataset_dir / "dataset_info.json"
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📄 Dataset metadata saved to: {metadata_file}")

def create_captions_file(dataset_dir: Path, captions: Dict[str, str], split_name: str):
    """Create captions file for training."""
    captions_file = dataset_dir / f"{split_name}_captions.txt"
    
    with open(captions_file, 'w') as f:
        for filename, caption in captions.items():
            f.write(f"{filename}: {caption}\n")
    
    return captions_file

# -------- 3. MAIN PROCESSING FUNCTION --------

def prepare_dataset(args):
    """Main dataset preparation function."""
    print(f"🎨 CompI Phase 1.E: Preparing LoRA Dataset for '{args.style_name}'")
    print("=" * 60)
    
    # Setup paths
    input_dir = Path(args.input_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("datasets") / args.style_name
    
    trigger_word = args.trigger_word or args.style_name
    
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Style name: {args.style_name}")
    print(f"🔤 Trigger word: {trigger_word}")
    print(f"📐 Target size: {args.size}x{args.size}")
    
    # Validate input directory
    print(f"\n🔍 Validating input directory...")
    valid_images, invalid_files = validate_image_directory(str(input_dir))
    
    print(f"✅ Found {len(valid_images)} valid images")
    if invalid_files:
        print(f"⚠️  Found {len(invalid_files)} invalid files:")
        for invalid in invalid_files[:5]:  # Show first 5
            print(f"   - {invalid}")
        if len(invalid_files) > 5:
            print(f"   ... and {len(invalid_files) - 5} more")
    
    if len(valid_images) < MIN_IMAGES_RECOMMENDED:
        print(f"⚠️  Warning: Only {len(valid_images)} images found. Recommended minimum: {MIN_IMAGES_RECOMMENDED}")
        print("   Consider adding more images for better style learning.")
    
    if args.validate_only:
        print("✅ Validation complete (--validate-only specified)")
        return
    
    # Create dataset structure
    print(f"\n📁 Creating dataset structure...")
    dataset_dir = create_dataset_structure(str(output_dir), args.style_name)
    
    # Split dataset
    train_files, val_files = split_dataset(valid_images, args.train_split)
    print(f"📊 Dataset split: {len(train_files)} train, {len(val_files)} validation")
    
    # Generate captions
    print(f"\n📝 Generating captions...")
    all_captions = generate_captions(valid_images, args.caption_template, trigger_word)
    
    # Process images
    print(f"\n🖼️  Processing images...")
    processed_count = 0
    processing_stats = []
    
    for split_name, file_list in [("train", train_files), ("validation", val_files)]:
        if not file_list:
            continue
            
        split_dir = dataset_dir / split_name
        split_captions = {}
        
        for filename in file_list:
            input_path = input_dir / filename
            output_filename = f"{Path(filename).stem}.png"
            output_path = split_dir / output_filename
            
            try:
                stats = process_image(str(input_path), str(output_path), args.size)
                processing_stats.append(stats)
                split_captions[output_filename] = all_captions[filename]
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"   Processed {processed_count}/{len(valid_images)} images...")
                    
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
        
        # Create captions file for this split
        if split_captions:
            captions_file = create_captions_file(dataset_dir, split_captions, split_name)
            print(f"📝 Created {split_name} captions: {captions_file}")
    
    # Save metadata
    metadata = {
        'style_name': args.style_name,
        'trigger_word': trigger_word,
        'total_images': len(valid_images),
        'train_images': len(train_files),
        'validation_images': len(val_files),
        'image_size': args.size,
        'caption_template': args.caption_template,
        'created_at': pd.Timestamp.now().isoformat(),
        'processing_stats': {
            'processed_count': processed_count,
            'failed_count': len(valid_images) - processed_count
        }
    }
    
    save_metadata(dataset_dir, metadata)
    
    print(f"\n🎉 Dataset preparation complete!")
    print(f"📁 Dataset location: {dataset_dir}")
    print(f"📊 Ready for LoRA training with {processed_count} processed images")
    print(f"\n💡 Next steps:")
    print(f"   1. Review the generated dataset in: {dataset_dir}")
    print(f"   2. Run LoRA training: python src/generators/compi_phase1e_lora_training.py --dataset-dir {dataset_dir}")

def main():
    """Main function."""
    args = setup_args()
    
    try:
        prepare_dataset(args)
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

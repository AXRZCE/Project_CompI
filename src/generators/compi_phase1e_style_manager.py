#!/usr/bin/env python3
"""
CompI Phase 1.E: LoRA Style Management System

Manage multiple LoRA styles, switch between them, and organize trained models.

Usage:
    python src/generators/compi_phase1e_style_manager.py --list
    python src/generators/compi_phase1e_style_manager.py --info my_style
    python src/generators/compi_phase1e_style_manager.py --cleanup
"""

import os
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# -------- 1. CONFIGURATION --------

LORA_MODELS_DIR = "lora_models"
STYLES_CONFIG_FILE = "lora_styles_config.json"

# -------- 2. STYLE MANAGEMENT CLASS --------

class LoRAStyleManager:
    """Manager for LoRA styles and models."""
    
    def __init__(self, models_dir: str = LORA_MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.config_file = self.models_dir / STYLES_CONFIG_FILE
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load styles configuration."""
        if self.config_file.exists():
            with open(self.config_file) as f:
                return json.load(f)
        return {"styles": {}, "last_updated": datetime.now().isoformat()}
    
    def save_config(self):
        """Save styles configuration."""
        self.config["last_updated"] = datetime.now().isoformat()
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def scan_styles(self) -> Dict[str, Dict]:
        """Scan for available LoRA styles."""
        styles = {}
        
        for style_dir in self.models_dir.iterdir():
            if not style_dir.is_dir() or style_dir.name.startswith('.'):
                continue
            
            # Look for checkpoints
            checkpoints = list(style_dir.glob("checkpoint-*"))
            if not checkpoints:
                continue
            
            # Get latest checkpoint
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
            
            # Load training info
            info_file = style_dir / "training_info.json"
            if info_file.exists():
                with open(info_file) as f:
                    training_info = json.load(f)
            else:
                training_info = {}
            
            # Load dataset info if available
            dataset_info = {}
            for dataset_dir in [style_dir / "dataset", Path("datasets") / style_dir.name]:
                dataset_info_file = dataset_dir / "dataset_info.json"
                if dataset_info_file.exists():
                    with open(dataset_info_file) as f:
                        dataset_info = json.load(f)
                    break
            
            # Compile style information
            style_info = {
                "name": style_dir.name,
                "path": str(style_dir),
                "latest_checkpoint": str(latest_checkpoint),
                "checkpoints": [str(cp) for cp in checkpoints],
                "training_info": training_info,
                "dataset_info": dataset_info,
                "last_scanned": datetime.now().isoformat()
            }
            
            styles[style_dir.name] = style_info
        
        return styles
    
    def refresh_styles(self):
        """Refresh the styles database."""
        print("🔄 Scanning for LoRA styles...")
        scanned_styles = self.scan_styles()
        
        # Update config
        self.config["styles"] = scanned_styles
        self.save_config()
        
        print(f"✅ Found {len(scanned_styles)} LoRA style(s)")
        return scanned_styles
    
    def list_styles(self, detailed: bool = False) -> List[Dict]:
        """List available styles."""
        styles = self.config.get("styles", {})
        
        if not styles:
            styles = self.refresh_styles()
        
        if detailed:
            return list(styles.values())
        else:
            return [{"name": name, "checkpoints": len(info["checkpoints"])} 
                   for name, info in styles.items()]
    
    def get_style_info(self, style_name: str) -> Optional[Dict]:
        """Get detailed information about a specific style."""
        styles = self.config.get("styles", {})
        return styles.get(style_name)
    
    def get_best_checkpoint(self, style_name: str) -> Optional[str]:
        """Get the best checkpoint for a style."""
        style_info = self.get_style_info(style_name)
        if not style_info:
            return None
        
        # For now, return the latest checkpoint
        # Could be enhanced to track validation loss and return best performing
        return style_info.get("latest_checkpoint")
    
    def delete_style(self, style_name: str, confirm: bool = False) -> bool:
        """Delete a LoRA style."""
        if not confirm:
            print("⚠️  Use --confirm to actually delete the style")
            return False
        
        style_dir = self.models_dir / style_name
        if not style_dir.exists():
            print(f"❌ Style not found: {style_name}")
            return False
        
        try:
            shutil.rmtree(style_dir)
            
            # Remove from config
            if style_name in self.config.get("styles", {}):
                del self.config["styles"][style_name]
                self.save_config()
            
            print(f"✅ Deleted style: {style_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting style: {e}")
            return False
    
    def cleanup_checkpoints(self, style_name: str, keep_last: int = 3) -> int:
        """Clean up old checkpoints, keeping only the most recent ones."""
        style_dir = self.models_dir / style_name
        if not style_dir.exists():
            print(f"❌ Style not found: {style_name}")
            return 0
        
        checkpoints = list(style_dir.glob("checkpoint-*"))
        if len(checkpoints) <= keep_last:
            print(f"✅ No cleanup needed for {style_name} ({len(checkpoints)} checkpoints)")
            return 0
        
        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.name.split('-')[1]))
        
        # Remove old checkpoints
        to_remove = checkpoints[:-keep_last]
        removed_count = 0
        
        for checkpoint in to_remove:
            try:
                shutil.rmtree(checkpoint)
                removed_count += 1
            except Exception as e:
                print(f"⚠️  Failed to remove {checkpoint}: {e}")
        
        print(f"✅ Cleaned up {removed_count} old checkpoints for {style_name}")
        return removed_count
    
    def export_style_info(self, output_file: str = None) -> str:
        """Export styles information to CSV."""
        styles = self.list_styles(detailed=True)
        
        if not styles:
            print("❌ No styles found")
            return ""
        
        # Prepare data for CSV
        rows = []
        for style in styles:
            training_info = style.get("training_info", {})
            dataset_info = style.get("dataset_info", {})
            
            row = {
                "style_name": style["name"],
                "checkpoints": len(style["checkpoints"]),
                "latest_checkpoint": Path(style["latest_checkpoint"]).name,
                "total_steps": training_info.get("total_steps", "unknown"),
                "epochs": training_info.get("epochs", "unknown"),
                "learning_rate": training_info.get("learning_rate", "unknown"),
                "lora_rank": training_info.get("lora_rank", "unknown"),
                "dataset_images": dataset_info.get("total_images", "unknown"),
                "trigger_word": dataset_info.get("trigger_word", "unknown"),
                "last_scanned": style.get("last_scanned", "unknown")
            }
            rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        
        if output_file is None:
            output_file = f"lora_styles_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df.to_csv(output_file, index=False)
        print(f"📊 Exported styles info to: {output_file}")
        return output_file

# -------- 3. COMMAND LINE INTERFACE --------

def setup_args():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(
        description="CompI Phase 1.E: LoRA Style Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available styles
  python %(prog)s --list
  
  # Get detailed info about a specific style
  python %(prog)s --info my_style
  
  # Refresh styles database
  python %(prog)s --refresh
  
  # Clean up old checkpoints
  python %(prog)s --cleanup my_style --keep 2
  
  # Export styles information
  python %(prog)s --export styles_report.csv
        """
    )
    
    parser.add_argument("--list", action="store_true",
                       help="List all available LoRA styles")
    
    parser.add_argument("--list-detailed", action="store_true",
                       help="List styles with detailed information")
    
    parser.add_argument("--info", metavar="STYLE_NAME",
                       help="Show detailed information about a specific style")
    
    parser.add_argument("--refresh", action="store_true",
                       help="Refresh the styles database")
    
    parser.add_argument("--cleanup", metavar="STYLE_NAME",
                       help="Clean up old checkpoints for a style")
    
    parser.add_argument("--keep", type=int, default=3,
                       help="Number of recent checkpoints to keep during cleanup")
    
    parser.add_argument("--delete", metavar="STYLE_NAME",
                       help="Delete a LoRA style")
    
    parser.add_argument("--confirm", action="store_true",
                       help="Confirm destructive operations")
    
    parser.add_argument("--export", metavar="OUTPUT_FILE",
                       help="Export styles information to CSV")
    
    parser.add_argument("--models-dir", default=LORA_MODELS_DIR,
                       help=f"LoRA models directory (default: {LORA_MODELS_DIR})")
    
    return parser.parse_args()

def print_style_info(style_info: Dict):
    """Print detailed style information."""
    print(f"🎨 Style: {style_info['name']}")
    print("=" * 40)
    
    # Basic info
    print(f"📁 Path: {style_info['path']}")
    print(f"📊 Checkpoints: {len(style_info['checkpoints'])}")
    print(f"🏆 Latest: {Path(style_info['latest_checkpoint']).name}")
    
    # Training info
    training_info = style_info.get("training_info", {})
    if training_info:
        print(f"\n🚀 Training Information:")
        print(f"   Steps: {training_info.get('total_steps', 'unknown')}")
        print(f"   Epochs: {training_info.get('epochs', 'unknown')}")
        print(f"   Learning Rate: {training_info.get('learning_rate', 'unknown')}")
        print(f"   LoRA Rank: {training_info.get('lora_rank', 'unknown')}")
        print(f"   LoRA Alpha: {training_info.get('lora_alpha', 'unknown')}")
    
    # Dataset info
    dataset_info = style_info.get("dataset_info", {})
    if dataset_info:
        print(f"\n📊 Dataset Information:")
        print(f"   Total Images: {dataset_info.get('total_images', 'unknown')}")
        print(f"   Train Images: {dataset_info.get('train_images', 'unknown')}")
        print(f"   Validation Images: {dataset_info.get('validation_images', 'unknown')}")
        print(f"   Trigger Word: {dataset_info.get('trigger_word', 'unknown')}")
        print(f"   Image Size: {dataset_info.get('image_size', 'unknown')}")
    
    print(f"\n🕒 Last Scanned: {style_info.get('last_scanned', 'unknown')}")

def main():
    """Main function."""
    args = setup_args()
    
    # Initialize style manager
    manager = LoRAStyleManager(args.models_dir)
    
    print("🎨 CompI Phase 1.E: LoRA Style Manager")
    print("=" * 40)
    
    # Execute commands
    if args.refresh:
        manager.refresh_styles()
    
    elif args.list or args.list_detailed:
        styles = manager.list_styles(detailed=args.list_detailed)
        
        if not styles:
            print("❌ No LoRA styles found")
            print("💡 Train a style first using: python src/generators/compi_phase1e_lora_training.py")
        else:
            print(f"📋 Available LoRA Styles ({len(styles)}):")
            print("-" * 40)
            
            if args.list_detailed:
                for style in styles:
                    print_style_info(style)
                    print()
            else:
                for style in styles:
                    print(f"🎨 {style['name']} ({style['checkpoints']} checkpoints)")
    
    elif args.info:
        style_info = manager.get_style_info(args.info)
        if style_info:
            print_style_info(style_info)
        else:
            print(f"❌ Style not found: {args.info}")
            print("💡 Use --list to see available styles")
    
    elif args.cleanup:
        removed = manager.cleanup_checkpoints(args.cleanup, args.keep)
        if removed > 0:
            manager.refresh_styles()
    
    elif args.delete:
        manager.delete_style(args.delete, args.confirm)
        if args.confirm:
            manager.refresh_styles()
    
    elif args.export:
        manager.export_style_info(args.export)
    
    else:
        print("❓ No command specified. Use --help for usage information.")
        print("💡 Common commands:")
        print("   --list              List available styles")
        print("   --info STYLE_NAME   Show style details")
        print("   --refresh           Refresh styles database")
    
    return 0

if __name__ == "__main__":
    exit(main())

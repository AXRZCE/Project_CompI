"""
CompI Phase 2.A: Audio-to-Image Generation

This module implements multimodal AI art generation that combines:
- Text prompts with style and mood conditioning
- Audio analysis and feature extraction
- Audio-to-text captioning
- Intelligent prompt fusion for enhanced creativity

Features:
- Support for various audio formats (mp3, wav, flac, etc.)
- Real-time audio analysis with tempo, energy, and spectral features
- OpenAI Whisper integration for audio captioning
- Comprehensive metadata logging and filename conventions
- Batch processing capabilities
"""

import os
import sys
import torch
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

from src.utils.audio_utils import AudioProcessor, AudioCaptioner, MultimodalPromptFusion, AudioFeatures
from src.utils.logging_utils import setup_logger
from src.utils.file_utils import ensure_directory_exists, generate_filename

# Setup logging
logger = setup_logger(__name__)

class CompIPhase2AAudioToImage:
    """
    CompI Phase 2.A: Audio-to-Image Generation System
    
    Combines text prompts with audio analysis to generate contextually rich AI art
    """
    
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        device: str = "auto",
        output_dir: str = "outputs",
        whisper_model: str = "base"
    ):
        """
        Initialize the audio-to-image generation system
        
        Args:
            model_name: Stable Diffusion model to use
            device: Device for inference (auto, cpu, cuda)
            output_dir: Directory for saving generated images
            whisper_model: Whisper model size for audio captioning
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.output_dir = Path(output_dir)
        ensure_directory_exists(self.output_dir)
        
        # Initialize components
        self.pipe = None
        self.audio_processor = AudioProcessor()
        self.audio_captioner = AudioCaptioner(model_size=whisper_model, device=self.device)
        self.prompt_fusion = MultimodalPromptFusion()
        
        logger.info(f"Initialized CompI Phase 2.A on {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate device"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        
        return device
    
    def _load_pipeline(self):
        """Lazy load the Stable Diffusion pipeline"""
        if self.pipe is None:
            logger.info(f"Loading Stable Diffusion model: {self.model_name}")
            
            # Custom safety checker that allows creative content
            def dummy_safety_checker(images, **kwargs):
                return images, [False] * len(images)
            
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=dummy_safety_checker,
                requires_safety_checker=False
            )
            
            self.pipe = self.pipe.to(self.device)
            self.pipe.enable_attention_slicing()
            
            if self.device == "cuda":
                self.pipe.enable_model_cpu_offload()
            
            logger.info("Stable Diffusion pipeline loaded successfully")
    
    def analyze_audio(self, audio_path: str, include_caption: bool = True) -> Tuple[AudioFeatures, str]:
        """
        Comprehensive audio analysis
        
        Args:
            audio_path: Path to audio file
            include_caption: Whether to generate audio caption
            
        Returns:
            Tuple of (AudioFeatures, audio_caption)
        """
        logger.info(f"Analyzing audio: {audio_path}")
        
        # Extract audio features
        audio_features = self.audio_processor.analyze_audio_file(audio_path)
        
        # Generate audio caption if requested
        audio_caption = ""
        if include_caption:
            try:
                audio_caption = self.audio_captioner.caption_audio(audio_path)
            except Exception as e:
                logger.warning(f"Audio captioning failed: {e}")
                audio_caption = ""
        
        return audio_features, audio_caption
    
    def generate_image(
        self,
        text_prompt: str,
        style: str = "",
        mood: str = "",
        audio_path: Optional[str] = None,
        num_images: int = 1,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> List[Dict]:
        """
        Generate images with optional audio conditioning
        
        Args:
            text_prompt: Base text prompt
            style: Art style
            mood: Mood/atmosphere
            audio_path: Optional path to audio file for conditioning
            num_images: Number of images to generate
            height: Image height
            width: Image width
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale for generation
            seed: Random seed for reproducibility
            
        Returns:
            List of generation results with metadata
        """
        self._load_pipeline()
        
        # Analyze audio if provided
        audio_features = None
        audio_caption = ""
        if audio_path and os.path.exists(audio_path):
            audio_features, audio_caption = self.analyze_audio(audio_path)
        
        # Create enhanced prompt
        if audio_features:
            enhanced_prompt = self.prompt_fusion.fuse_prompt_with_audio(
                text_prompt, style, mood, audio_features, audio_caption
            )
        else:
            enhanced_prompt = text_prompt
            if style:
                enhanced_prompt += f", {style}"
            if mood:
                enhanced_prompt += f", {mood}"
        
        logger.info(f"Generating {num_images} image(s) with prompt: {enhanced_prompt}")
        
        results = []
        
        for i in range(num_images):
            # Set up generation parameters
            current_seed = seed if seed is not None else torch.seed()
            generator = torch.Generator(device=self.device).manual_seed(current_seed)
            
            # Generate image
            with torch.autocast(self.device) if self.device == "cuda" else torch.no_grad():
                result = self.pipe(
                    enhanced_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                )
                
                image = result.images[0]
            
            # Create metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "text_prompt": text_prompt,
                "style": style,
                "mood": mood,
                "enhanced_prompt": enhanced_prompt,
                "audio_path": audio_path,
                "audio_caption": audio_caption,
                "generation_params": {
                    "height": height,
                    "width": width,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": current_seed,
                    "model": self.model_name
                },
                "device": self.device,
                "phase": "2A_audio_to_image"
            }
            
            # Add audio features to metadata
            if audio_features:
                metadata["audio_features"] = audio_features.to_dict()
                metadata["audio_tags"] = self.prompt_fusion.generate_audio_tags(audio_features)
            
            # Generate filename
            filename = self._generate_filename(
                text_prompt, style, mood, current_seed, i + 1, 
                has_audio=audio_path is not None
            )
            
            # Save image and metadata
            image_path = self.output_dir / f"{filename}.png"
            metadata_path = self.output_dir / f"{filename}_metadata.json"
            
            image.save(image_path)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            results.append({
                "image": image,
                "image_path": str(image_path),
                "metadata_path": str(metadata_path),
                "metadata": metadata,
                "filename": filename
            })
            
            logger.info(f"Generated image {i+1}/{num_images}: {filename}")
        
        return results
    
    def _generate_filename(
        self, 
        prompt: str, 
        style: str, 
        mood: str, 
        seed: int, 
        variation: int,
        has_audio: bool = False
    ) -> str:
        """Generate descriptive filename following CompI conventions"""
        
        # Create prompt slug (first 5 words)
        prompt_words = prompt.lower().replace(',', '').split()[:5]
        prompt_slug = "_".join(prompt_words)
        
        # Create style and mood slugs
        style_slug = style.replace(" ", "").replace(",", "")[:10] if style else "standard"
        mood_slug = mood.replace(" ", "").replace(",", "")[:10] if mood else "neutral"
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Audio indicator
        audio_tag = "_AUDIO" if has_audio else ""
        
        # Combine all elements
        filename = f"{prompt_slug}_{style_slug}_{mood_slug}_{timestamp}_seed{seed}{audio_tag}_v{variation}"
        
        return filename
    
    def batch_process(
        self,
        audio_directory: str,
        text_prompt: str,
        style: str = "",
        mood: str = "",
        **generation_kwargs
    ) -> List[Dict]:
        """
        Process multiple audio files in batch
        
        Args:
            audio_directory: Directory containing audio files
            text_prompt: Base text prompt for all generations
            style: Art style
            mood: Mood/atmosphere
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of all generation results
        """
        audio_dir = Path(audio_directory)
        if not audio_dir.exists():
            raise ValueError(f"Audio directory not found: {audio_directory}")
        
        # Find audio files
        audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}
        audio_files = [
            f for f in audio_dir.iterdir() 
            if f.suffix.lower() in audio_extensions
        ]
        
        if not audio_files:
            raise ValueError(f"No audio files found in {audio_directory}")
        
        logger.info(f"Processing {len(audio_files)} audio files")
        
        all_results = []
        for audio_file in audio_files:
            logger.info(f"Processing: {audio_file.name}")
            
            try:
                results = self.generate_image(
                    text_prompt=text_prompt,
                    style=style,
                    mood=mood,
                    audio_path=str(audio_file),
                    **generation_kwargs
                )
                all_results.extend(results)
                
            except Exception as e:
                logger.error(f"Error processing {audio_file.name}: {e}")
                continue
        
        logger.info(f"Batch processing complete: {len(all_results)} images generated")
        return all_results

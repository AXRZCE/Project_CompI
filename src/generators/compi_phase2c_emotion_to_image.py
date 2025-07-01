"""
CompI Phase 2.C: Emotional/Contextual Input to Image Generation

This module implements emotion-driven AI art generation that combines:
- Emotion detection and sentiment analysis
- Contextual mood processing
- Emoji and text-based emotion recognition
- Color palette generation based on emotions
- Intelligent fusion of emotional context with creative prompts

Features:
- Support for preset emotions, custom emotions, and emoji input
- Automatic sentiment analysis with TextBlob
- Emotion-to-color palette mapping
- Contextual prompt enhancement
- Comprehensive metadata logging and filename conventions
- Batch processing capabilities for multiple emotional contexts
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

from src.utils.emotion_utils import EmotionProcessor, EmotionalPromptEnhancer, EmotionAnalysis, EmotionCategory
from src.utils.logging_utils import setup_logger
from src.utils.file_utils import ensure_directory_exists, generate_filename

# Setup logging
logger = setup_logger(__name__)

class CompIPhase2CEmotionToImage:
    """
    CompI Phase 2.C: Emotional/Contextual Input to Image Generation System
    
    Transforms emotions, moods, and contextual feelings into AI-generated art
    """
    
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        device: str = "auto",
        output_dir: str = "outputs"
    ):
        """
        Initialize the emotion-to-image generation system
        
        Args:
            model_name: Stable Diffusion model to use
            device: Device for inference (auto, cpu, cuda)
            output_dir: Directory for saving generated images
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.output_dir = Path(output_dir)
        ensure_directory_exists(self.output_dir)
        
        # Initialize components
        self.pipe = None
        self.emotion_processor = EmotionProcessor()
        self.prompt_enhancer = EmotionalPromptEnhancer()
        
        logger.info(f"Initialized CompI Phase 2.C on {self.device}")
    
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
    
    def analyze_emotion(
        self,
        emotion_input: str,
        emotion_type: str = "auto",
        contextual_text: Optional[str] = None
    ) -> EmotionAnalysis:
        """
        Comprehensive emotion analysis
        
        Args:
            emotion_input: Emotion input (preset, custom, emoji, or text)
            emotion_type: Type of input ('preset', 'custom', 'emoji', 'text', 'auto')
            contextual_text: Additional contextual text for analysis
            
        Returns:
            EmotionAnalysis object with complete analysis
        """
        logger.info(f"Analyzing emotion input: {emotion_input}")
        
        # Combine inputs for analysis
        analysis_text = emotion_input
        if contextual_text:
            analysis_text += f" {contextual_text}"
        
        # Determine selected emotion for preset types
        selected_emotion = None
        if emotion_type == "preset" or (emotion_type == "auto" and emotion_input.lower() in self.emotion_processor.preset_emotions):
            selected_emotion = emotion_input.lower()
        
        # Perform emotion analysis
        emotion_analysis = self.emotion_processor.analyze_emotion(analysis_text, selected_emotion)
        
        return emotion_analysis
    
    def generate_image(
        self,
        text_prompt: str,
        style: str = "",
        emotion_input: str = "",
        emotion_type: str = "auto",
        contextual_text: str = "",
        enhancement_strength: float = 0.7,
        num_images: int = 1,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> List[Dict]:
        """
        Generate images with emotional conditioning
        
        Args:
            text_prompt: Base text prompt
            style: Art style
            emotion_input: Emotion input (preset, custom, emoji, or descriptive text)
            emotion_type: Type of emotion input
            contextual_text: Additional contextual description
            enhancement_strength: How strongly to apply emotion (0-1)
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
        
        # Analyze emotion if provided
        emotion_analysis = None
        if emotion_input.strip():
            emotion_analysis = self.analyze_emotion(emotion_input, emotion_type, contextual_text)
        
        # Create enhanced prompt
        if emotion_analysis:
            enhanced_prompt = self.prompt_enhancer.enhance_prompt_with_emotion(
                text_prompt, style, emotion_analysis, enhancement_strength
            )
        else:
            enhanced_prompt = text_prompt
            if style:
                enhanced_prompt += f", {style}"
        
        logger.info(f"Generating {num_images} image(s) with enhanced prompt")
        
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
                "emotion_input": emotion_input,
                "emotion_type": emotion_type,
                "contextual_text": contextual_text,
                "enhancement_strength": enhancement_strength,
                "enhanced_prompt": enhanced_prompt,
                "generation_params": {
                    "height": height,
                    "width": width,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": current_seed,
                    "model": self.model_name
                },
                "device": self.device,
                "phase": "2C_emotion_to_image"
            }
            
            # Add emotion analysis to metadata
            if emotion_analysis:
                metadata["emotion_analysis"] = emotion_analysis.to_dict()
                metadata["emotion_tags"] = self.prompt_enhancer.generate_emotion_tags(emotion_analysis)
            
            # Generate filename
            filename = self._generate_filename(
                text_prompt, style, emotion_analysis, current_seed, i + 1
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
                "filename": filename,
                "emotion_analysis": emotion_analysis
            })
            
            logger.info(f"Generated image {i+1}/{num_images}: {filename}")
        
        return results

    def _generate_filename(
        self,
        prompt: str,
        style: str,
        emotion_analysis: Optional[EmotionAnalysis],
        seed: int,
        variation: int
    ) -> str:
        """Generate descriptive filename following CompI conventions"""

        # Create prompt slug (first 5 words)
        prompt_words = prompt.lower().replace(',', '').split()[:5]
        prompt_slug = "_".join(prompt_words)

        # Create style slug
        style_slug = style.replace(" ", "").replace(",", "")[:10] if style else "standard"

        # Create emotion slug
        if emotion_analysis:
            emotion_slug = f"{emotion_analysis.primary_emotion.value}_{emotion_analysis.intensity_level}"[:15]
        else:
            emotion_slug = "neutral"

        # Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Combine all elements
        filename = f"{prompt_slug}_{style_slug}_{emotion_slug}_{timestamp}_seed{seed}_EMO_v{variation}"

        return filename

    def batch_process_emotions(
        self,
        text_prompt: str,
        style: str,
        emotions: List[str],
        emotion_type: str = "auto",
        **generation_kwargs
    ) -> List[Dict]:
        """
        Process multiple emotions in batch

        Args:
            text_prompt: Base text prompt for all generations
            style: Art style
            emotions: List of emotions to process
            emotion_type: Type of emotion input
            **generation_kwargs: Additional generation parameters

        Returns:
            List of all generation results
        """
        logger.info(f"Processing {len(emotions)} emotions in batch")

        all_results = []
        for i, emotion in enumerate(emotions):
            logger.info(f"Processing emotion {i+1}/{len(emotions)}: {emotion}")

            try:
                results = self.generate_image(
                    text_prompt=text_prompt,
                    style=style,
                    emotion_input=emotion,
                    emotion_type=emotion_type,
                    **generation_kwargs
                )
                all_results.extend(results)

            except Exception as e:
                logger.error(f"Error processing emotion '{emotion}': {e}")
                continue

        logger.info(f"Batch processing complete: {len(all_results)} images generated")
        return all_results

    def generate_emotion_palette_art(
        self,
        text_prompt: str,
        style: str,
        emotion_input: str,
        use_color_conditioning: bool = True,
        **generation_kwargs
    ) -> List[Dict]:
        """
        Generate art using emotion-derived color palettes

        Args:
            text_prompt: Base text prompt
            style: Art style
            emotion_input: Emotion input
            use_color_conditioning: Whether to add color palette to prompt
            **generation_kwargs: Additional generation parameters

        Returns:
            List of generation results with color palette conditioning
        """
        # Analyze emotion to get color palette
        emotion_analysis = self.analyze_emotion(emotion_input)

        # Enhance prompt with color information if requested
        if use_color_conditioning and emotion_analysis:
            color_names = self._hex_to_color_names(emotion_analysis.color_palette)
            color_prompt = f"with a color palette of {', '.join(color_names)}"
            enhanced_text_prompt = f"{text_prompt}, {color_prompt}"
        else:
            enhanced_text_prompt = text_prompt

        return self.generate_image(
            text_prompt=enhanced_text_prompt,
            style=style,
            emotion_input=emotion_input,
            **generation_kwargs
        )

    def _hex_to_color_names(self, hex_colors: List[str]) -> List[str]:
        """Convert hex colors to approximate color names"""
        color_mapping = {
            "#FFD700": "golden", "#FFA500": "orange", "#FF69B4": "pink",
            "#00CED1": "turquoise", "#32CD32": "lime", "#4169E1": "blue",
            "#6495ED": "cornflower", "#708090": "slate", "#2F4F4F": "dark slate",
            "#191970": "midnight blue", "#DC143C": "crimson", "#B22222": "firebrick",
            "#8B0000": "dark red", "#FF4500": "orange red", "#FF6347": "tomato",
            "#800080": "purple", "#4B0082": "indigo", "#2E2E2E": "dark gray",
            "#696969": "dim gray", "#A9A9A9": "dark gray", "#FF1493": "deep pink",
            "#FFB6C1": "light pink", "#FFC0CB": "pink", "#FFFF00": "yellow",
            "#C71585": "medium violet", "#DB7093": "pale violet", "#20B2AA": "light sea green",
            "#48D1CC": "medium turquoise", "#40E0D0": "turquoise", "#AFEEEE": "pale turquoise",
            "#9370DB": "medium purple", "#8A2BE2": "blue violet", "#7B68EE": "medium slate blue",
            "#6A5ACD": "slate blue", "#483D8B": "dark slate blue", "#808080": "gray",
            "#C0C0C0": "silver", "#D3D3D3": "light gray", "#DCDCDC": "gainsboro"
        }

        color_names = []
        for hex_color in hex_colors:
            color_name = color_mapping.get(hex_color.upper(), "colorful")
            color_names.append(color_name)

        return color_names

"""
CompI Phase 2.D: Real-Time Data Feeds to Image Generation

This module implements real-time data-driven AI art generation that combines:
- Weather data integration from multiple APIs
- News headlines and RSS feed processing
- Financial market data incorporation
- Real-time context analysis and summarization
- Intelligent fusion of real-time data with creative prompts

Features:
- Support for weather, news, and financial data feeds
- Automatic data caching and rate limiting
- Context-aware prompt enhancement
- Temporal and thematic analysis
- Comprehensive metadata logging and filename conventions
- Batch processing capabilities for multiple data sources
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

from src.utils.realtime_data_utils import (
    RealTimeDataProcessor, RealTimeContext, DataFeedType, RealTimeDataPoint
)
from src.utils.logging_utils import setup_logger
from src.utils.file_utils import ensure_directory_exists, generate_filename

# Setup logging
logger = setup_logger(__name__)

class CompIPhase2DRealTimeToImage:
    """
    CompI Phase 2.D: Real-Time Data Feeds to Image Generation System
    
    Transforms real-time data feeds into AI-generated art
    """
    
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        device: str = "auto",
        output_dir: str = "outputs"
    ):
        """
        Initialize the real-time data-to-image generation system
        
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
        self.data_processor = RealTimeDataProcessor()
        
        logger.info(f"Initialized CompI Phase 2.D on {self.device}")
    
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
    
    def fetch_realtime_context(
        self,
        include_weather: bool = False,
        weather_city: str = "New York",
        weather_api_key: Optional[str] = None,
        include_news: bool = False,
        news_category: str = "general",
        max_news: int = 3,
        news_api_key: Optional[str] = None,
        include_financial: bool = False
    ) -> RealTimeContext:
        """
        Fetch real-time context from various data sources
        
        Args:
            include_weather: Whether to include weather data
            weather_city: City for weather data
            weather_api_key: Optional weather API key
            include_news: Whether to include news data
            news_category: Category of news to fetch
            max_news: Maximum number of news items
            news_api_key: Optional news API key
            include_financial: Whether to include financial data
            
        Returns:
            RealTimeContext with processed data
        """
        logger.info("Fetching real-time context for art generation")
        
        return self.data_processor.fetch_realtime_context(
            include_weather=include_weather,
            weather_city=weather_city,
            include_news=include_news,
            news_category=news_category,
            max_news=max_news,
            include_financial=include_financial,
            weather_api_key=weather_api_key,
            news_api_key=news_api_key
        )
    
    def generate_image(
        self,
        text_prompt: str,
        style: str = "",
        mood: str = "",
        include_weather: bool = False,
        weather_city: str = "New York",
        weather_api_key: Optional[str] = None,
        include_news: bool = False,
        news_category: str = "general",
        max_news: int = 3,
        news_api_key: Optional[str] = None,
        include_financial: bool = False,
        context_strength: float = 0.7,
        num_images: int = 1,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> List[Dict]:
        """
        Generate images with real-time data conditioning
        
        Args:
            text_prompt: Base text prompt
            style: Art style
            mood: Mood/atmosphere
            include_weather: Whether to include weather data
            weather_city: City for weather data
            weather_api_key: Optional weather API key
            include_news: Whether to include news data
            news_category: Category of news to fetch
            max_news: Maximum number of news items
            news_api_key: Optional news API key
            include_financial: Whether to include financial data
            context_strength: How strongly to apply real-time context (0-1)
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
        
        # Fetch real-time context if any data sources are enabled
        realtime_context = None
        if include_weather or include_news or include_financial:
            realtime_context = self.fetch_realtime_context(
                include_weather=include_weather,
                weather_city=weather_city,
                weather_api_key=weather_api_key,
                include_news=include_news,
                news_category=news_category,
                max_news=max_news,
                news_api_key=news_api_key,
                include_financial=include_financial
            )
        
        # Create enhanced prompt
        enhanced_prompt = self._create_enhanced_prompt(
            text_prompt, style, mood, realtime_context, context_strength
        )
        
        logger.info(f"Generating {num_images} image(s) with real-time context")
        
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
                "context_strength": context_strength,
                "data_sources": {
                    "weather": include_weather,
                    "news": include_news,
                    "financial": include_financial
                },
                "generation_params": {
                    "height": height,
                    "width": width,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": current_seed,
                    "model": self.model_name
                },
                "device": self.device,
                "phase": "2D_realtime_to_image"
            }
            
            # Add real-time context to metadata
            if realtime_context:
                metadata["realtime_context"] = realtime_context.to_dict()
            
            # Generate filename
            filename = self._generate_filename(
                text_prompt, style, realtime_context, current_seed, i + 1
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
                "realtime_context": realtime_context
            })
            
            logger.info(f"Generated image {i+1}/{num_images}: {filename}")
        
        return results

    def _create_enhanced_prompt(
        self,
        text_prompt: str,
        style: str,
        mood: str,
        realtime_context: Optional[RealTimeContext],
        context_strength: float
    ) -> str:
        """
        Create enhanced prompt with real-time context

        Args:
            text_prompt: Base text prompt
            style: Art style
            mood: Mood/atmosphere
            realtime_context: Real-time context data
            context_strength: How strongly to apply context (0-1)

        Returns:
            Enhanced prompt with real-time context
        """
        enhanced_prompt = text_prompt.strip()

        # Add style
        if style:
            enhanced_prompt += f", {style}"

        # Add mood
        if mood:
            enhanced_prompt += f", {mood}"

        # Add real-time context based on strength
        if realtime_context and context_strength > 0:
            if context_strength > 0.7:
                # Strong context integration
                enhanced_prompt += f", {realtime_context.artistic_inspiration}"
                if realtime_context.mood_indicators:
                    mood_text = ", ".join(realtime_context.mood_indicators[:2])
                    enhanced_prompt += f", with {mood_text} influences"

            elif context_strength > 0.4:
                # Moderate context integration
                enhanced_prompt += f", {realtime_context.artistic_inspiration}"

            else:
                # Subtle context integration
                if realtime_context.key_themes:
                    theme = realtime_context.key_themes[0]
                    enhanced_prompt += f", inspired by {theme}"

        return enhanced_prompt

    def _generate_filename(
        self,
        prompt: str,
        style: str,
        realtime_context: Optional[RealTimeContext],
        seed: int,
        variation: int
    ) -> str:
        """Generate descriptive filename following CompI conventions"""

        # Create prompt slug (first 5 words)
        prompt_words = prompt.lower().replace(',', '').split()[:5]
        prompt_slug = "_".join(prompt_words)

        # Create style slug
        style_slug = style.replace(" ", "").replace(",", "")[:10] if style else "standard"

        # Create context slug
        if realtime_context and realtime_context.data_points:
            context_types = []
            for dp in realtime_context.data_points:
                context_types.append(dp.feed_type.value[:3])  # First 3 chars
            context_slug = "_".join(set(context_types))[:15]
        else:
            context_slug = "static"

        # Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Combine all elements
        filename = f"{prompt_slug}_{style_slug}_{context_slug}_{timestamp}_seed{seed}_RTDATA_v{variation}"

        return filename

    def batch_process_data_sources(
        self,
        text_prompt: str,
        style: str,
        data_source_configs: List[Dict],
        **generation_kwargs
    ) -> List[Dict]:
        """
        Process multiple data source configurations in batch

        Args:
            text_prompt: Base text prompt for all generations
            style: Art style
            data_source_configs: List of data source configuration dictionaries
            **generation_kwargs: Additional generation parameters

        Returns:
            List of all generation results
        """
        logger.info(f"Processing {len(data_source_configs)} data source configurations")

        all_results = []
        for i, config in enumerate(data_source_configs):
            logger.info(f"Processing configuration {i+1}/{len(data_source_configs)}")

            try:
                results = self.generate_image(
                    text_prompt=text_prompt,
                    style=style,
                    **config,
                    **generation_kwargs
                )
                all_results.extend(results)

            except Exception as e:
                logger.error(f"Error processing configuration {i+1}: {e}")
                continue

        logger.info(f"Batch processing complete: {len(all_results)} images generated")
        return all_results

    def generate_temporal_series(
        self,
        text_prompt: str,
        style: str,
        data_config: Dict,
        time_intervals: List[int],
        **generation_kwargs
    ) -> List[Dict]:
        """
        Generate a series of images with real-time data at different time intervals

        Args:
            text_prompt: Base text prompt
            style: Art style
            data_config: Data source configuration
            time_intervals: List of time intervals in minutes between generations
            **generation_kwargs: Additional generation parameters

        Returns:
            List of generation results across time
        """
        import time

        logger.info(f"Generating temporal series with {len(time_intervals)} intervals")

        all_results = []

        for i, interval in enumerate(time_intervals):
            if i > 0:  # Don't wait before first generation
                logger.info(f"Waiting {interval} minutes before next generation...")
                time.sleep(interval * 60)  # Convert minutes to seconds

            logger.info(f"Generating image {i+1}/{len(time_intervals)}")

            try:
                # Clear cache to ensure fresh data
                self.data_processor.cache.cache.clear()

                results = self.generate_image(
                    text_prompt=text_prompt,
                    style=style,
                    **data_config,
                    **generation_kwargs
                )

                # Add temporal metadata
                for result in results:
                    result["metadata"]["temporal_series"] = {
                        "series_index": i,
                        "total_in_series": len(time_intervals),
                        "interval_minutes": interval if i > 0 else 0
                    }

                all_results.extend(results)

            except Exception as e:
                logger.error(f"Error in temporal generation {i+1}: {e}")
                continue

        logger.info(f"Temporal series complete: {len(all_results)} images generated")
        return all_results

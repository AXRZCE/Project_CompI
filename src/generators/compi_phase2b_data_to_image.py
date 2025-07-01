"""
CompI Phase 2.B: Data/Logic Input to Image Generation

This module implements data-driven AI art generation that combines:
- CSV data analysis and processing
- Mathematical formula evaluation
- Data-to-text conversion for prompt enhancement
- Data visualization for artistic conditioning
- Intelligent fusion of data insights with creative prompts

Features:
- Support for CSV files with comprehensive data analysis
- Safe mathematical formula evaluation with NumPy
- Poetic text generation from data patterns
- Data visualization creation for artistic inspiration
- Comprehensive metadata logging and filename conventions
- Batch processing capabilities for multiple datasets
"""

import os
import sys
import torch
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from diffusers import StableDiffusionPipeline
from PIL import Image

from src.utils.data_utils import DataProcessor, DataToTextConverter, DataVisualizer, DataFeatures
from src.utils.logging_utils import setup_logger
from src.utils.file_utils import ensure_directory_exists, generate_filename

# Setup logging
logger = setup_logger(__name__)

class CompIPhase2BDataToImage:
    """
    CompI Phase 2.B: Data/Logic Input to Image Generation System
    
    Transforms structured data and mathematical formulas into AI-generated art
    """
    
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        device: str = "auto",
        output_dir: str = "outputs",
        visualization_style: str = "artistic"
    ):
        """
        Initialize the data-to-image generation system
        
        Args:
            model_name: Stable Diffusion model to use
            device: Device for inference (auto, cpu, cuda)
            output_dir: Directory for saving generated images
            visualization_style: Style for data visualizations
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.output_dir = Path(output_dir)
        ensure_directory_exists(self.output_dir)
        
        # Initialize components
        self.pipe = None
        self.data_processor = DataProcessor()
        self.text_converter = DataToTextConverter()
        self.visualizer = DataVisualizer(style=visualization_style)
        
        logger.info(f"Initialized CompI Phase 2.B on {self.device}")
    
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
    
    def analyze_csv_data(self, csv_path: str) -> Tuple[pd.DataFrame, DataFeatures, str, Image.Image]:
        """
        Comprehensive CSV data analysis
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Tuple of (DataFrame, DataFeatures, poetic_description, visualization_image)
        """
        logger.info(f"Analyzing CSV data: {csv_path}")
        
        # Load and analyze data
        df = pd.read_csv(csv_path)
        features = self.data_processor.analyze_csv_data(df)
        
        # Generate poetic description
        poetic_description = self.text_converter.generate_poetic_description(features)
        
        # Create visualization
        visualization_image = self.visualizer.create_data_visualization(df, features)
        
        return df, features, poetic_description, visualization_image
    
    def evaluate_mathematical_formula(self, formula: str, num_points: int = 100) -> Tuple[np.ndarray, Dict, str, Image.Image]:
        """
        Evaluate mathematical formula and create artistic interpretation
        
        Args:
            formula: Mathematical expression
            num_points: Number of points to generate
            
        Returns:
            Tuple of (result_array, metadata, poetic_description, visualization_image)
        """
        logger.info(f"Evaluating mathematical formula: {formula}")
        
        # Evaluate formula
        result_array, metadata = self.data_processor.evaluate_formula(formula, num_points)
        
        # Generate poetic description
        poetic_description = self.text_converter.generate_formula_description(formula, metadata)
        
        # Create visualization
        visualization_image = self.visualizer.create_formula_visualization(result_array, formula, metadata)
        
        return result_array, metadata, poetic_description, visualization_image

    def generate_image(
        self,
        text_prompt: str,
        style: str = "",
        mood: str = "",
        csv_path: Optional[str] = None,
        formula: Optional[str] = None,
        num_images: int = 1,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        save_data_visualization: bool = True
    ) -> List[Dict]:
        """
        Generate images with data/formula conditioning

        Args:
            text_prompt: Base text prompt
            style: Art style
            mood: Mood/atmosphere
            csv_path: Optional path to CSV file
            formula: Optional mathematical formula
            num_images: Number of images to generate
            height: Image height
            width: Image width
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale for generation
            seed: Random seed for reproducibility
            save_data_visualization: Whether to save data visualization

        Returns:
            List of generation results with metadata
        """
        self._load_pipeline()

        # Process data input
        data_features = None
        poetic_description = ""
        data_visualization = None
        data_type = "none"

        if csv_path and os.path.exists(csv_path):
            df, data_features, poetic_description, data_visualization = self.analyze_csv_data(csv_path)
            data_type = "csv"
        elif formula and formula.strip():
            result_array, formula_metadata, poetic_description, data_visualization = self.evaluate_mathematical_formula(formula)
            data_type = "formula"

        # Create enhanced prompt
        enhanced_prompt = text_prompt
        if style:
            enhanced_prompt += f", {style}"
        if mood:
            enhanced_prompt += f", {mood}"
        if poetic_description:
            enhanced_prompt += f", {poetic_description}"

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
                "mood": mood,
                "enhanced_prompt": enhanced_prompt,
                "poetic_description": poetic_description,
                "data_type": data_type,
                "csv_path": csv_path,
                "formula": formula,
                "generation_params": {
                    "height": height,
                    "width": width,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": current_seed,
                    "model": self.model_name
                },
                "device": self.device,
                "phase": "2B_data_to_image"
            }

            # Add data features to metadata
            if data_features:
                metadata["data_features"] = data_features.to_dict()

            # Generate filename
            filename = self._generate_filename(
                text_prompt, style, mood, current_seed, i + 1,
                data_type=data_type
            )

            # Save image and metadata
            image_path = self.output_dir / f"{filename}.png"
            metadata_path = self.output_dir / f"{filename}_metadata.json"

            image.save(image_path)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Save data visualization if requested
            data_viz_path = None
            if save_data_visualization and data_visualization:
                data_viz_path = self.output_dir / f"{filename}_data_viz.png"
                data_visualization.save(data_viz_path)

            results.append({
                "image": image,
                "image_path": str(image_path),
                "metadata_path": str(metadata_path),
                "data_visualization_path": str(data_viz_path) if data_viz_path else None,
                "data_visualization": data_visualization,
                "metadata": metadata,
                "filename": filename,
                "poetic_description": poetic_description
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
        data_type: str = "none"
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

        # Data type indicator
        data_tag = f"_{data_type.upper()}" if data_type != "none" else ""

        # Combine all elements
        filename = f"{prompt_slug}_{style_slug}_{mood_slug}_{timestamp}_seed{seed}{data_tag}_v{variation}"

        return filename

    def batch_process_csv_files(
        self,
        csv_directory: str,
        text_prompt: str,
        style: str = "",
        mood: str = "",
        **generation_kwargs
    ) -> List[Dict]:
        """
        Process multiple CSV files in batch

        Args:
            csv_directory: Directory containing CSV files
            text_prompt: Base text prompt for all generations
            style: Art style
            mood: Mood/atmosphere
            **generation_kwargs: Additional generation parameters

        Returns:
            List of all generation results
        """
        csv_dir = Path(csv_directory)
        if not csv_dir.exists():
            raise ValueError(f"CSV directory not found: {csv_directory}")

        # Find CSV files
        csv_files = list(csv_dir.glob("*.csv"))

        if not csv_files:
            raise ValueError(f"No CSV files found in {csv_directory}")

        logger.info(f"Processing {len(csv_files)} CSV files")

        all_results = []
        for csv_file in csv_files:
            logger.info(f"Processing: {csv_file.name}")

            try:
                results = self.generate_image(
                    text_prompt=text_prompt,
                    style=style,
                    mood=mood,
                    csv_path=str(csv_file),
                    **generation_kwargs
                )
                all_results.extend(results)

            except Exception as e:
                logger.error(f"Error processing {csv_file.name}: {e}")
                continue

        logger.info(f"Batch processing complete: {len(all_results)} images generated")
        return all_results

    def batch_process_formulas(
        self,
        formulas: List[str],
        text_prompt: str,
        style: str = "",
        mood: str = "",
        **generation_kwargs
    ) -> List[Dict]:
        """
        Process multiple mathematical formulas in batch

        Args:
            formulas: List of mathematical formulas
            text_prompt: Base text prompt for all generations
            style: Art style
            mood: Mood/atmosphere
            **generation_kwargs: Additional generation parameters

        Returns:
            List of all generation results
        """
        logger.info(f"Processing {len(formulas)} mathematical formulas")

        all_results = []
        for i, formula in enumerate(formulas):
            logger.info(f"Processing formula {i+1}/{len(formulas)}: {formula}")

            try:
                results = self.generate_image(
                    text_prompt=text_prompt,
                    style=style,
                    mood=mood,
                    formula=formula,
                    **generation_kwargs
                )
                all_results.extend(results)

            except Exception as e:
                logger.error(f"Error processing formula '{formula}': {e}")
                continue

        logger.info(f"Batch processing complete: {len(all_results)} images generated")
        return all_results

"""
CompI Data Processing Utilities

This module provides utilities for Phase 2.B: Data/Logic Input Integration
- CSV data analysis and processing
- Mathematical formula evaluation
- Data-to-text conversion (poetic descriptions)
- Data visualization generation
- Statistical analysis and pattern detection
"""

import os
import io
import ast
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from PIL import Image
import logging

logger = logging.getLogger(__name__)

@dataclass
class DataFeatures:
    """Container for extracted data features and statistics"""
    
    # Basic properties
    shape: Tuple[int, int]
    columns: List[str]
    numeric_columns: List[str]
    data_types: Dict[str, str]
    
    # Statistical features
    means: Dict[str, float]
    medians: Dict[str, float]
    stds: Dict[str, float]
    mins: Dict[str, float]
    maxs: Dict[str, float]
    ranges: Dict[str, float]
    
    # Pattern features
    trends: Dict[str, str]  # 'increasing', 'decreasing', 'stable', 'volatile'
    correlations: Dict[str, float]  # strongest correlations
    seasonality: Dict[str, bool]  # detected patterns
    
    # Derived insights
    complexity_score: float  # 0-1 measure of data complexity
    variability_score: float  # 0-1 measure of data variability
    pattern_strength: float  # 0-1 measure of detectable patterns
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'shape': self.shape,
            'columns': self.columns,
            'numeric_columns': self.numeric_columns,
            'data_types': self.data_types,
            'means': self.means,
            'medians': self.medians,
            'stds': self.stds,
            'mins': self.mins,
            'maxs': self.maxs,
            'ranges': self.ranges,
            'trends': self.trends,
            'correlations': self.correlations,
            'seasonality': self.seasonality,
            'complexity_score': self.complexity_score,
            'variability_score': self.variability_score,
            'pattern_strength': self.pattern_strength
        }

class DataProcessor:
    """Core data processing and analysis functionality"""
    
    def __init__(self):
        """Initialize the data processor"""
        self.safe_functions = {
            # Math functions
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'len': len, 'pow': pow,
            
            # NumPy functions
            'np': np, 'numpy': np,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
            'pi': np.pi, 'e': np.e,
            
            # Math module functions
            'math': math,
            
            # Restricted builtins
            '__builtins__': {}
        }
    
    def analyze_csv_data(self, df: pd.DataFrame) -> DataFeatures:
        """
        Comprehensive analysis of CSV data
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFeatures object with extracted insights
        """
        logger.info(f"Analyzing CSV data with shape {df.shape}")
        
        # Basic properties
        shape = df.shape
        columns = df.columns.tolist()
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_columns = numeric_df.columns.tolist()
        data_types = {col: str(df[col].dtype) for col in columns}
        
        # Statistical features
        means = {col: float(numeric_df[col].mean()) for col in numeric_columns}
        medians = {col: float(numeric_df[col].median()) for col in numeric_columns}
        stds = {col: float(numeric_df[col].std()) for col in numeric_columns}
        mins = {col: float(numeric_df[col].min()) for col in numeric_columns}
        maxs = {col: float(numeric_df[col].max()) for col in numeric_columns}
        ranges = {col: maxs[col] - mins[col] for col in numeric_columns}
        
        # Pattern analysis
        trends = self._analyze_trends(numeric_df)
        correlations = self._find_strongest_correlations(numeric_df)
        seasonality = self._detect_seasonality(numeric_df)
        
        # Derived scores
        complexity_score = self._calculate_complexity_score(numeric_df)
        variability_score = self._calculate_variability_score(stds, ranges)
        pattern_strength = self._calculate_pattern_strength(trends, correlations)
        
        return DataFeatures(
            shape=shape,
            columns=columns,
            numeric_columns=numeric_columns,
            data_types=data_types,
            means=means,
            medians=medians,
            stds=stds,
            mins=mins,
            maxs=maxs,
            ranges=ranges,
            trends=trends,
            correlations=correlations,
            seasonality=seasonality,
            complexity_score=complexity_score,
            variability_score=variability_score,
            pattern_strength=pattern_strength
        )
    
    def evaluate_formula(self, formula: str, num_points: int = 100) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Safely evaluate mathematical formula
        
        Args:
            formula: Mathematical expression (Python/NumPy syntax)
            num_points: Number of points to generate
            
        Returns:
            Tuple of (result_array, metadata)
        """
        logger.info(f"Evaluating formula: {formula}")
        
        try:
            # Create default x values if not specified in formula
            if 'x' in formula and 'linspace' not in formula and 'arange' not in formula:
                # Add default x range if x is used but not defined
                x = np.linspace(0, 10, num_points)
                self.safe_functions['x'] = x
            
            # Evaluate the formula
            result = eval(formula, self.safe_functions)
            
            # Ensure result is a numpy array
            if not isinstance(result, np.ndarray):
                if isinstance(result, (list, tuple)):
                    result = np.array(result)
                else:
                    # Single value - create array
                    result = np.full(num_points, result)
            
            # Analyze the result
            metadata = {
                'length': len(result),
                'min': float(np.min(result)),
                'max': float(np.max(result)),
                'mean': float(np.mean(result)),
                'std': float(np.std(result)),
                'range': float(np.max(result) - np.min(result)),
                'formula': formula,
                'has_pattern': self._detect_mathematical_pattern(result)
            }
            
            return result, metadata
            
        except Exception as e:
            logger.error(f"Formula evaluation failed: {e}")
            raise ValueError(f"Invalid formula: {e}")

    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analyze trends in numeric columns"""
        trends = {}
        for col in df.columns:
            values = df[col].dropna()
            if len(values) < 3:
                trends[col] = 'insufficient_data'
                continue

            # Calculate trend using linear regression slope
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            std_val = values.std()

            if abs(slope) < std_val * 0.1:
                trends[col] = 'stable'
            elif std_val > values.mean() * 0.5:
                trends[col] = 'volatile'
            elif slope > 0:
                trends[col] = 'increasing'
            else:
                trends[col] = 'decreasing'

        return trends

    def _find_strongest_correlations(self, df: pd.DataFrame) -> Dict[str, float]:
        """Find strongest correlations between columns"""
        if len(df.columns) < 2:
            return {}

        corr_matrix = df.corr()
        correlations = {}

        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns):
                if i < j:  # Avoid duplicates and self-correlation
                    corr_val = corr_matrix.loc[col1, col2]
                    if not np.isnan(corr_val):
                        correlations[f"{col1}_vs_{col2}"] = float(corr_val)

        # Return top 3 strongest correlations
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        return dict(sorted_corr[:3])

    def _detect_seasonality(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Simple seasonality detection"""
        seasonality = {}
        for col in df.columns:
            values = df[col].dropna()
            if len(values) < 12:  # Need at least 12 points for seasonality
                seasonality[col] = False
                continue

            # Simple autocorrelation check
            try:
                autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
                seasonality[col] = not np.isnan(autocorr) and abs(autocorr) > 0.3
            except:
                seasonality[col] = False

        return seasonality

    def _calculate_complexity_score(self, df: pd.DataFrame) -> float:
        """Calculate data complexity score (0-1)"""
        if df.empty:
            return 0.0

        # Factors: number of columns, data types variety, missing values
        num_cols = len(df.columns)
        col_score = min(num_cols / 10, 1.0)  # Normalize to 0-1

        # Missing data complexity
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        missing_score = min(missing_ratio * 2, 1.0)

        return (col_score + missing_score) / 2

    def _calculate_variability_score(self, stds: Dict[str, float], ranges: Dict[str, float]) -> float:
        """Calculate data variability score (0-1)"""
        if not stds:
            return 0.0

        # Normalize standard deviations by their ranges
        normalized_vars = []
        for col in stds:
            if ranges[col] > 0:
                normalized_vars.append(stds[col] / ranges[col])

        if not normalized_vars:
            return 0.0

        return min(np.mean(normalized_vars) * 2, 1.0)

    def _calculate_pattern_strength(self, trends: Dict[str, str], correlations: Dict[str, float]) -> float:
        """Calculate pattern strength score (0-1)"""
        pattern_score = 0.0

        # Trend strength
        trend_patterns = sum(1 for trend in trends.values() if trend in ['increasing', 'decreasing'])
        trend_score = min(trend_patterns / max(len(trends), 1), 1.0)

        # Correlation strength
        if correlations:
            max_corr = max(abs(corr) for corr in correlations.values())
            corr_score = max_corr
        else:
            corr_score = 0.0

        return (trend_score + corr_score) / 2

    def _detect_mathematical_pattern(self, data: np.ndarray) -> bool:
        """Detect if mathematical data has recognizable patterns"""
        if len(data) < 10:
            return False

        # Check for periodicity using autocorrelation
        try:
            # Simple pattern detection
            autocorr = np.corrcoef(data[:-1], data[1:])[0, 1]
            return not np.isnan(autocorr) and abs(autocorr) > 0.5
        except:
            return False


class DataToTextConverter:
    """Convert data patterns into poetic/narrative text descriptions"""

    def __init__(self):
        """Initialize the converter with descriptive vocabularies"""
        self.trend_descriptions = {
            'increasing': ['ascending', 'rising', 'climbing', 'growing', 'soaring'],
            'decreasing': ['descending', 'falling', 'declining', 'diminishing', 'fading'],
            'stable': ['steady', 'constant', 'balanced', 'harmonious', 'peaceful'],
            'volatile': ['chaotic', 'turbulent', 'dynamic', 'energetic', 'wild']
        }

        self.pattern_adjectives = {
            'high_complexity': ['intricate', 'complex', 'sophisticated', 'elaborate'],
            'low_complexity': ['simple', 'pure', 'minimal', 'clean'],
            'high_variability': ['diverse', 'varied', 'rich', 'multifaceted'],
            'low_variability': ['consistent', 'uniform', 'regular', 'predictable'],
            'strong_patterns': ['rhythmic', 'structured', 'organized', 'patterned'],
            'weak_patterns': ['random', 'scattered', 'free-flowing', 'organic']
        }

        self.artistic_metaphors = [
            'like brushstrokes on a canvas',
            'resembling musical notes in harmony',
            'flowing like water through landscapes',
            'dancing with mathematical precision',
            'weaving patterns of light and shadow',
            'creating symphonies of numbers',
            'painting stories with data points',
            'sculpting meaning from statistics'
        ]

    def generate_poetic_description(self, features: DataFeatures) -> str:
        """
        Generate poetic description from data features

        Args:
            features: DataFeatures object

        Returns:
            Poetic text description
        """
        descriptions = []

        # Basic data description
        descriptions.append(f"A tapestry woven from {features.shape[0]} data points across {features.shape[1]} dimensions")

        # Trend descriptions
        trend_desc = self._describe_trends(features.trends)
        if trend_desc:
            descriptions.append(trend_desc)

        # Variability description
        var_desc = self._describe_variability(features.variability_score)
        if var_desc:
            descriptions.append(var_desc)

        # Pattern description
        pattern_desc = self._describe_patterns(features.pattern_strength, features.correlations)
        if pattern_desc:
            descriptions.append(pattern_desc)

        # Add artistic metaphor
        import random
        metaphor = random.choice(self.artistic_metaphors)
        descriptions.append(f"The data flows {metaphor}")

        return '. '.join(descriptions) + '.'

    def generate_formula_description(self, formula: str, metadata: Dict[str, Any]) -> str:
        """
        Generate poetic description for mathematical formula

        Args:
            formula: Original formula
            metadata: Formula evaluation metadata

        Returns:
            Poetic text description
        """
        descriptions = []

        # Formula introduction
        descriptions.append(f"Mathematical harmony emerges from the expression: {formula}")

        # Range description
        range_val = metadata['range']
        if range_val > 10:
            descriptions.append("The function soars across vast numerical landscapes")
        elif range_val > 1:
            descriptions.append("Values dance within moderate bounds")
        else:
            descriptions.append("Numbers whisper in gentle, subtle variations")

        # Pattern description
        if metadata['has_pattern']:
            descriptions.append("Revealing intricate patterns that speak to the soul")
        else:
            descriptions.append("Creating unique, unrepeatable mathematical poetry")

        # Add artistic metaphor
        import random
        metaphor = random.choice(self.artistic_metaphors)
        descriptions.append(f"Each calculation {metaphor}")

        return '. '.join(descriptions) + '.'

    def _describe_trends(self, trends: Dict[str, str]) -> str:
        """Describe overall trends in the data"""
        if not trends:
            return ""

        trend_counts = {}
        for trend in trends.values():
            trend_counts[trend] = trend_counts.get(trend, 0) + 1

        dominant_trend = max(trend_counts, key=trend_counts.get)

        if dominant_trend in self.trend_descriptions:
            import random
            adj = random.choice(self.trend_descriptions[dominant_trend])
            return f"The data reveals {adj} patterns throughout its structure"

        return ""

    def _describe_variability(self, variability_score: float) -> str:
        """Describe data variability"""
        import random

        if variability_score > 0.7:
            adj = random.choice(self.pattern_adjectives['high_variability'])
            return f"With {adj} expressions of numerical diversity"
        elif variability_score < 0.3:
            adj = random.choice(self.pattern_adjectives['low_variability'])
            return f"Maintaining {adj} elegance in its values"
        else:
            return "Balancing consistency with creative variation"

    def _describe_patterns(self, pattern_strength: float, correlations: Dict[str, float]) -> str:
        """Describe pattern strength and correlations"""
        import random

        if pattern_strength > 0.6:
            adj = random.choice(self.pattern_adjectives['strong_patterns'])
            return f"Displaying {adj} relationships between its elements"
        elif pattern_strength < 0.3:
            adj = random.choice(self.pattern_adjectives['weak_patterns'])
            return f"Embracing {adj} freedom in its numerical expression"
        else:
            return "Weaving subtle connections throughout its numerical fabric"


class DataVisualizer:
    """Create visualizations from data for artistic conditioning"""

    def __init__(self, style: str = 'artistic'):
        """
        Initialize visualizer

        Args:
            style: Visualization style ('artistic', 'scientific', 'minimal')
        """
        self.style = style
        self.color_palettes = {
            'artistic': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
            'scientific': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83'],
            'minimal': ['#2C3E50', '#34495E', '#7F8C8D', '#95A5A6', '#BDC3C7']
        }

    def create_data_visualization(self, df: pd.DataFrame, features: DataFeatures) -> Image.Image:
        """
        Create artistic visualization from DataFrame

        Args:
            df: Input DataFrame
            features: DataFeatures object

        Returns:
            PIL Image of the visualization
        """
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Data Pattern Visualization', fontsize=16, fontweight='bold')

        numeric_df = df.select_dtypes(include=[np.number])
        colors = self.color_palettes[self.style]

        # Plot 1: Line plot of first few columns
        ax1 = axes[0, 0]
        for i, col in enumerate(numeric_df.columns[:3]):
            ax1.plot(numeric_df[col], color=colors[i % len(colors)],
                    linewidth=2, alpha=0.8, label=col)
        ax1.set_title('Data Trends', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Distribution/histogram
        ax2 = axes[0, 1]
        if len(numeric_df.columns) > 0:
            col = numeric_df.columns[0]
            ax2.hist(numeric_df[col].dropna(), bins=20, color=colors[0],
                    alpha=0.7, edgecolor='black')
            ax2.set_title(f'Distribution: {col}', fontweight='bold')
            ax2.grid(True, alpha=0.3)

        # Plot 3: Correlation heatmap (if multiple columns)
        ax3 = axes[1, 0]
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            im = ax3.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            ax3.set_xticks(range(len(corr_matrix.columns)))
            ax3.set_yticks(range(len(corr_matrix.columns)))
            ax3.set_xticklabels(corr_matrix.columns, rotation=45)
            ax3.set_yticklabels(corr_matrix.columns)
            ax3.set_title('Correlations', fontweight='bold')
            plt.colorbar(im, ax=ax3, shrink=0.8)
        else:
            ax3.text(0.5, 0.5, 'Single Column\nNo Correlations',
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Correlations', fontweight='bold')

        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        if len(numeric_df.columns) > 0:
            stats_data = [features.means[col] for col in numeric_df.columns[:5]]
            bars = ax4.bar(range(len(stats_data)), stats_data, color=colors[:len(stats_data)])
            ax4.set_title('Mean Values', fontweight='bold')
            ax4.set_xticks(range(len(stats_data)))
            ax4.set_xticklabels([col[:8] for col in numeric_df.columns[:5]], rotation=45)
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)

        return Image.open(buf)

    def create_formula_visualization(self, data: np.ndarray, formula: str, metadata: Dict[str, Any]) -> Image.Image:
        """
        Create artistic visualization from formula result

        Args:
            data: Formula result array
            formula: Original formula
            metadata: Formula metadata

        Returns:
            PIL Image of the visualization
        """
        try:
            logger.info(f"Creating visualization for formula: {formula}")
            logger.info(f"Data shape: {data.shape}, Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")

            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Mathematical Pattern: {formula}', fontsize=14, fontweight='bold')

        colors = self.color_palettes[self.style]
        x = np.arange(len(data))

        # Plot 1: Main function plot
        ax1 = axes[0, 0]
        ax1.plot(x, data, color=colors[0], linewidth=3, alpha=0.8)
        ax1.fill_between(x, data, alpha=0.3, color=colors[0])
        ax1.set_title('Function Values', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Derivative approximation
        ax2 = axes[0, 1]
        if len(data) > 1:
            derivative = np.gradient(data)
            ax2.plot(x, derivative, color=colors[1], linewidth=2)
            ax2.set_title('Rate of Change', fontweight='bold')
            ax2.grid(True, alpha=0.3)

        # Plot 3: Distribution
        ax3 = axes[1, 0]
        ax3.hist(data, bins=30, color=colors[2], alpha=0.7, edgecolor='black')
        ax3.set_title('Value Distribution', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Phase space (if applicable)
        ax4 = axes[1, 1]
        if len(data) > 1:
            ax4.scatter(data[:-1], data[1:], c=x[:-1], cmap='viridis', alpha=0.6)
            ax4.set_xlabel('f(t)')
            ax4.set_ylabel('f(t+1)')
            ax4.set_title('Phase Space', fontweight='bold')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            # Convert to PIL Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            buf.seek(0)

            image = Image.open(buf)
            logger.info(f"Successfully created visualization image: {image.size}")
            return image

        except Exception as e:
            logger.error(f"Error creating formula visualization: {e}")
            plt.close('all')  # Clean up any open figures

            # Return a simple error image
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'Visualization Error:\n{str(e)}',
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            buf.seek(0)

            return Image.open(buf)

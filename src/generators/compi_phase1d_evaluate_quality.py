#!/usr/bin/env python3
"""
CompI Phase 1.D: Baseline Output Quality Evaluation Tool

This tool provides systematic evaluation of generated images with:
- Visual quality assessment
- Prompt adherence scoring
- Style/mood consistency evaluation
- Objective metrics calculation
- Comprehensive logging and tracking

Usage:
    python src/generators/compi_phase1d_evaluate_quality.py
    # Or via wrapper: python run_evaluation.py
"""

import os
import re
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

import streamlit as st
from PIL import Image
import imagehash
import pandas as pd

# -------- 1. CONFIGURATION --------

OUTPUT_DIR = "outputs"
EVAL_CSV = "outputs/evaluation_log.csv"
EVAL_SUMMARY = "outputs/evaluation_summary.json"

# Filename patterns for different CompI phases
FILENAME_PATTERNS = [
    # Phase 1.B Advanced styling: prompt_style_mood_timestamp_seed_variation
    re.compile(r"^(?P<prompt>[a-z0-9_,]+)_(?P<style>[a-zA-Z0-9]+)_(?P<mood>[a-zA-Z0-9]+)_(?P<timestamp>\d{8}_\d{6})_seed(?P<seed>\d+)_v(?P<variation>\d+)\.png$"),
    # Phase 1.A Basic generation: prompt_timestamp_seed
    re.compile(r"^(?P<prompt>[a-z0-9_,]+)_(?P<timestamp>\d{8}_\d{6})_seed(?P<seed>\d+)\.png$"),
    # Alternative pattern: prompt_style_timestamp_seed
    re.compile(r"^(?P<prompt>[a-z0-9_,]+)_(?P<style>[a-zA-Z0-9]+)_(?P<timestamp>\d{8}_\d{6})_seed(?P<seed>\d+)\.png$"),
]

# Evaluation criteria
EVALUATION_CRITERIA = {
    "prompt_match": {
        "name": "Prompt Adherence",
        "description": "How well does the image match the text prompt?",
        "scale": "1=Poor match, 3=Good match, 5=Perfect match"
    },
    "style_consistency": {
        "name": "Style Consistency", 
        "description": "How well does the image reflect the intended artistic style?",
        "scale": "1=Style not evident, 3=Style present, 5=Style perfectly executed"
    },
    "mood_atmosphere": {
        "name": "Mood & Atmosphere",
        "description": "How well does the image convey the intended mood/atmosphere?", 
        "scale": "1=Wrong mood, 3=Neutral/adequate, 5=Perfect mood"
    },
    "technical_quality": {
        "name": "Technical Quality",
        "description": "Overall image quality (resolution, composition, artifacts)",
        "scale": "1=Poor quality, 3=Acceptable, 5=Excellent quality"
    },
    "creative_appeal": {
        "name": "Creative Appeal",
        "description": "Subjective aesthetic and creative value",
        "scale": "1=Unappealing, 3=Decent, 5=Highly appealing"
    }
}

# -------- 2. UTILITY FUNCTIONS --------

def parse_filename(filename: str) -> Optional[Dict]:
    """Parse filename to extract metadata using multiple patterns."""
    for pattern in FILENAME_PATTERNS:
        match = pattern.match(filename)
        if match:
            data = match.groupdict()
            data["filename"] = filename
            # Set defaults for missing fields
            data.setdefault("style", "unknown")
            data.setdefault("mood", "unknown") 
            data.setdefault("variation", "1")
            return data
    return None

def get_image_metrics(image_path: str) -> Dict:
    """Calculate objective image metrics."""
    try:
        img = Image.open(image_path)
        file_size = os.path.getsize(image_path)
        
        # Perceptual hashes for similarity detection
        phash = str(imagehash.phash(img))
        dhash = str(imagehash.dhash(img))
        
        # Basic image stats
        width, height = img.size
        aspect_ratio = width / height
        
        # Color analysis
        if img.mode == 'RGB':
            colors = img.getcolors(maxcolors=256*256*256)
            unique_colors = len(colors) if colors else 0
        else:
            unique_colors = 0
            
        return {
            "width": width,
            "height": height,
            "aspect_ratio": round(aspect_ratio, 3),
            "file_size_kb": round(file_size / 1024, 2),
            "unique_colors": unique_colors,
            "phash": phash,
            "dhash": dhash,
            "format": img.format,
            "mode": img.mode
        }
    except Exception as e:
        return {"error": str(e)}

def load_existing_evaluations() -> Dict:
    """Load existing evaluations from CSV."""
    if not os.path.exists(EVAL_CSV):
        return {}
    
    try:
        df = pd.read_csv(EVAL_CSV)
        return df.set_index('filename').to_dict('index')
    except Exception:
        return {}

def save_evaluation(filename: str, metadata: Dict, scores: Dict, notes: str, metrics: Dict):
    """Save evaluation to CSV file."""
    # Prepare row data
    row_data = {
        "filename": filename,
        "timestamp": datetime.now().isoformat(),
        "prompt": metadata.get("prompt", ""),
        "style": metadata.get("style", ""),
        "mood": metadata.get("mood", ""),
        "seed": metadata.get("seed", ""),
        "variation": metadata.get("variation", ""),
        "generation_timestamp": metadata.get("timestamp", ""),
        "notes": notes,
        **scores,  # Add all evaluation scores
        **{f"metric_{k}": v for k, v in metrics.items() if k != "error"}  # Add metrics with prefix
    }
    
    # Create CSV if it doesn't exist
    file_exists = os.path.exists(EVAL_CSV)
    
    with open(EVAL_CSV, "a", newline='', encoding='utf-8') as f:
        fieldnames = list(row_data.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

# -------- 3. STREAMLIT UI --------

def main():
    st.set_page_config(
        page_title="CompI - Quality Evaluation", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🕵️ CompI Phase 1.D: Baseline Output Quality Evaluation")
    
    st.markdown("""
    **Systematic evaluation tool for CompI-generated images**
    
    This tool helps you:
    - 📊 Assess image quality across multiple criteria
    - 📈 Track improvements over time  
    - 🔍 Calculate objective metrics
    - 📝 Maintain detailed evaluation logs
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Output directory selection
        output_dir = st.text_input("Output Directory", OUTPUT_DIR)
        
        # Evaluation mode
        eval_mode = st.selectbox(
            "Evaluation Mode",
            ["Single Image Review", "Batch Evaluation", "Summary Analysis"]
        )
        
        # Filter options
        st.subheader("🔍 Filters")
        show_evaluated = st.checkbox("Show already evaluated", True)
        show_unevaluated = st.checkbox("Show unevaluated", True)
    
    # Load images
    if not os.path.exists(output_dir):
        st.error(f"Output directory '{output_dir}' not found!")
        return
        
    image_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.png')]
    parsed_images = []
    
    for fname in image_files:
        metadata = parse_filename(fname)
        if metadata:
            parsed_images.append(metadata)
    
    if not parsed_images:
        st.warning("No CompI-generated images found with recognizable filename patterns.")
        st.info("Expected patterns: prompt_style_mood_timestamp_seed_variation.png")
        return
    
    # Load existing evaluations
    existing_evals = load_existing_evaluations()
    
    # Filter images based on evaluation status
    filtered_images = []
    for img_data in parsed_images:
        fname = img_data["filename"]
        is_evaluated = fname in existing_evals
        
        if (show_evaluated and is_evaluated) or (show_unevaluated and not is_evaluated):
            img_data["is_evaluated"] = is_evaluated
            filtered_images.append(img_data)
    
    st.info(f"Found {len(filtered_images)} images matching your filters")
    
    # Main evaluation interface
    if eval_mode == "Single Image Review":
        single_image_evaluation(filtered_images, existing_evals, output_dir)
    elif eval_mode == "Batch Evaluation":
        batch_evaluation(filtered_images, existing_evals, output_dir)
    else:
        summary_analysis(existing_evals)

def single_image_evaluation(images: List[Dict], existing_evals: Dict, output_dir: str):
    """Single image evaluation interface."""
    if not images:
        st.warning("No images available for evaluation.")
        return
    
    # Image selection
    image_options = [f"{img['filename']} {'✅' if img['is_evaluated'] else '❌'}" for img in images]
    selected_idx = st.selectbox("Select Image to Evaluate", range(len(image_options)), format_func=lambda x: image_options[x])
    
    if selected_idx is None:
        return
        
    img_data = images[selected_idx]
    fname = img_data["filename"]
    img_path = os.path.join(output_dir, fname)
    
    # Display image and metadata
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖼️ Image")
        try:
            image = Image.open(img_path)
            st.image(image, use_column_width=True)
            
            # Calculate metrics
            metrics = get_image_metrics(img_path)
            if "error" not in metrics:
                st.subheader("📊 Objective Metrics")
                st.json(metrics)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return
    
    with col2:
        st.subheader("📋 Metadata")
        st.json({k: v for k, v in img_data.items() if k != "filename"})
        
        # Evaluation form
        st.subheader("⭐ Evaluation")
        
        # Load existing scores if available
        existing = existing_evals.get(fname, {})
        
        with st.form(f"eval_form_{fname}"):
            scores = {}
            for criterion_key, criterion_info in EVALUATION_CRITERIA.items():
                scores[criterion_key] = st.slider(
                    f"{criterion_info['name']}",
                    min_value=1, max_value=5, 
                    value=int(existing.get(criterion_key, 3)),
                    help=f"{criterion_info['description']}\n{criterion_info['scale']}"
                )
            
            notes = st.text_area(
                "Notes & Comments",
                value=existing.get("notes", ""),
                help="Additional observations, issues, or suggestions"
            )
            
            submitted = st.form_submit_button("💾 Save Evaluation")
            
            if submitted:
                save_evaluation(fname, img_data, scores, notes, metrics)
                st.success(f"✅ Evaluation saved for {fname}")
                st.experimental_rerun()

def batch_evaluation(images: List[Dict], existing_evals: Dict, output_dir: str):
    """Batch evaluation interface for multiple images."""
    st.subheader("📦 Batch Evaluation")

    unevaluated = [img for img in images if not img['is_evaluated']]

    if not unevaluated:
        st.info("All images have been evaluated!")
        return

    st.info(f"{len(unevaluated)} images pending evaluation")

    # Quick batch scoring
    with st.form("batch_eval_form"):
        st.write("**Quick Batch Scoring** (applies to all unevaluated images)")

        batch_scores = {}
        for criterion_key, criterion_info in EVALUATION_CRITERIA.items():
            batch_scores[criterion_key] = st.slider(
                f"Default {criterion_info['name']}",
                min_value=1, max_value=5, value=3,
                help=f"Default score for {criterion_info['description']}"
            )

        batch_notes = st.text_area("Default Notes", "Batch evaluation")

        if st.form_submit_button("Apply to All Unevaluated"):
            progress_bar = st.progress(0)

            for i, img_data in enumerate(unevaluated):
                fname = img_data["filename"]
                img_path = os.path.join(output_dir, fname)
                metrics = get_image_metrics(img_path)

                save_evaluation(fname, img_data, batch_scores, batch_notes, metrics)
                progress_bar.progress((i + 1) / len(unevaluated))

            st.success(f"✅ Batch evaluation completed for {len(unevaluated)} images!")
            st.experimental_rerun()

def summary_analysis(existing_evals: Dict):
    """Display evaluation summary and analytics."""
    st.subheader("📈 Evaluation Summary & Analytics")

    if not existing_evals:
        st.warning("No evaluations found. Please evaluate some images first.")
        return

    # Convert to DataFrame for analysis
    df = pd.DataFrame.from_dict(existing_evals, orient='index')

    # Basic statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Evaluated", len(df))

    with col2:
        if 'prompt_match' in df.columns:
            avg_prompt_match = df['prompt_match'].mean()
            st.metric("Avg Prompt Match", f"{avg_prompt_match:.2f}/5")

    with col3:
        if 'technical_quality' in df.columns:
            avg_quality = df['technical_quality'].mean()
            st.metric("Avg Technical Quality", f"{avg_quality:.2f}/5")

    # Detailed analytics
    st.subheader("📊 Detailed Analytics")

    # Score distribution
    if any(col in df.columns for col in EVALUATION_CRITERIA.keys()):
        st.write("**Score Distribution by Criteria**")

        score_cols = [col for col in EVALUATION_CRITERIA.keys() if col in df.columns]
        if score_cols:
            score_data = df[score_cols].mean().sort_values(ascending=False)
            st.bar_chart(score_data)

    # Style/Mood analysis
    if 'style' in df.columns and 'mood' in df.columns:
        st.write("**Performance by Style & Mood**")

        col1, col2 = st.columns(2)

        with col1:
            if 'prompt_match' in df.columns:
                style_performance = df.groupby('style')['prompt_match'].mean().sort_values(ascending=False)
                st.write("**Best Performing Styles (Prompt Match)**")
                st.bar_chart(style_performance)

        with col2:
            if 'creative_appeal' in df.columns:
                mood_performance = df.groupby('mood')['creative_appeal'].mean().sort_values(ascending=False)
                st.write("**Best Performing Moods (Creative Appeal)**")
                st.bar_chart(mood_performance)

    # Recent evaluations
    st.subheader("🕒 Recent Evaluations")

    if 'timestamp' in df.columns:
        recent_df = df.sort_values('timestamp', ascending=False).head(10)
        display_cols = ['prompt', 'style', 'mood'] + [col for col in EVALUATION_CRITERIA.keys() if col in df.columns]
        display_cols = [col for col in display_cols if col in recent_df.columns]

        if display_cols:
            st.dataframe(recent_df[display_cols])

    # Export options
    st.subheader("💾 Export Data")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Download CSV"):
            csv_data = df.to_csv()
            st.download_button(
                label="Download Evaluation Data",
                data=csv_data,
                file_name=f"compi_evaluation_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("📋 Generate Report"):
            # Generate summary report
            report = generate_evaluation_report(df)
            st.text_area("Evaluation Report", report, height=300)

def generate_evaluation_report(df: pd.DataFrame) -> str:
    """Generate a text summary report of evaluations."""
    report_lines = [
        "# CompI Phase 1.D - Evaluation Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary Statistics",
        f"- Total Images Evaluated: {len(df)}",
    ]

    # Add score summaries
    for criterion_key, criterion_info in EVALUATION_CRITERIA.items():
        if criterion_key in df.columns:
            mean_score = df[criterion_key].mean()
            std_score = df[criterion_key].std()
            report_lines.append(f"- {criterion_info['name']}: {mean_score:.2f} ± {std_score:.2f}")

    # Add style/mood analysis
    if 'style' in df.columns:
        report_lines.extend([
            "",
            "## Style Performance",
        ])

        if 'prompt_match' in df.columns:
            style_scores = df.groupby('style')['prompt_match'].mean().sort_values(ascending=False)
            for style, score in style_scores.head(5).items():
                report_lines.append(f"- {style}: {score:.2f}")

    if 'mood' in df.columns:
        report_lines.extend([
            "",
            "## Mood Performance",
        ])

        if 'creative_appeal' in df.columns:
            mood_scores = df.groupby('mood')['creative_appeal'].mean().sort_values(ascending=False)
            for mood, score in mood_scores.head(5).items():
                report_lines.append(f"- {mood}: {score:.2f}")

    # Add recommendations
    report_lines.extend([
        "",
        "## Recommendations",
        "- Focus on improving lowest-scoring criteria",
        "- Experiment with best-performing style/mood combinations",
        "- Consider adjusting generation parameters for consistency",
        "- Continue systematic evaluation for trend analysis"
    ])

    return "\n".join(report_lines)

if __name__ == "__main__":
    main()

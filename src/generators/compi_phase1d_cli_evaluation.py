#!/usr/bin/env python3
"""
CompI Phase 1.D: Command-Line Quality Evaluation Tool

Command-line interface for batch evaluation and analysis of generated images.

Usage:
    python src/generators/compi_phase1d_cli_evaluation.py --help
    python src/generators/compi_phase1d_cli_evaluation.py --analyze
    python src/generators/compi_phase1d_cli_evaluation.py --batch-score 4 3 4 4 3
"""

import os
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image

# Import functions from the main evaluation module
from compi_phase1d_evaluate_quality import (
    parse_filename, get_image_metrics, load_existing_evaluations,
    save_evaluation, EVALUATION_CRITERIA, OUTPUT_DIR, EVAL_CSV
)

def setup_args():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(
        description="CompI Phase 1.D: Command-Line Quality Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze existing evaluations
  python %(prog)s --analyze
  
  # Batch score all unevaluated images (prompt_match, style, mood, quality, appeal)
  python %(prog)s --batch-score 4 3 4 4 3 --notes "Batch evaluation - good quality"
  
  # Generate detailed report
  python %(prog)s --report --output evaluation_report.txt
  
  # List unevaluated images
  python %(prog)s --list-unevaluated
        """
    )
    
    parser.add_argument("--output-dir", default=OUTPUT_DIR, 
                       help="Directory containing generated images")
    
    # Analysis commands
    parser.add_argument("--analyze", action="store_true",
                       help="Display evaluation summary and statistics")
    
    parser.add_argument("--report", action="store_true",
                       help="Generate detailed evaluation report")
    
    parser.add_argument("--output", "-o", 
                       help="Output file for report (default: stdout)")
    
    # Batch evaluation
    parser.add_argument("--batch-score", nargs=5, type=int, metavar=("PROMPT", "STYLE", "MOOD", "QUALITY", "APPEAL"),
                       help="Batch score all unevaluated images (1-5 for each criteria)")
    
    parser.add_argument("--notes", default="CLI batch evaluation",
                       help="Notes for batch evaluation")
    
    # Listing commands
    parser.add_argument("--list-all", action="store_true",
                       help="List all images with evaluation status")
    
    parser.add_argument("--list-evaluated", action="store_true", 
                       help="List only evaluated images")
    
    parser.add_argument("--list-unevaluated", action="store_true",
                       help="List only unevaluated images")
    
    # Filtering
    parser.add_argument("--style", help="Filter by style")
    parser.add_argument("--mood", help="Filter by mood")
    
    return parser.parse_args()

def load_images(output_dir: str) -> List[Dict]:
    """Load and parse all images from output directory."""
    if not os.path.exists(output_dir):
        print(f"❌ Output directory '{output_dir}' not found!")
        return []
    
    image_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.png')]
    parsed_images = []
    
    for fname in image_files:
        metadata = parse_filename(fname)
        if metadata:
            parsed_images.append(metadata)
    
    return parsed_images

def filter_images(images: List[Dict], style: str = None, mood: str = None) -> List[Dict]:
    """Filter images by style and/or mood."""
    filtered = images
    
    if style:
        filtered = [img for img in filtered if img.get('style', '').lower() == style.lower()]
    
    if mood:
        filtered = [img for img in filtered if img.get('mood', '').lower() == mood.lower()]
    
    return filtered

def analyze_evaluations(existing_evals: Dict):
    """Display evaluation analysis."""
    if not existing_evals:
        print("❌ No evaluations found.")
        return
    
    df = pd.DataFrame.from_dict(existing_evals, orient='index')
    
    print("📊 CompI Phase 1.D - Evaluation Analysis")
    print("=" * 50)
    print(f"Total Evaluated Images: {len(df)}")
    print()
    
    # Score statistics
    print("📈 Score Statistics:")
    for criterion_key, criterion_info in EVALUATION_CRITERIA.items():
        if criterion_key in df.columns:
            mean_score = df[criterion_key].mean()
            std_score = df[criterion_key].std()
            min_score = df[criterion_key].min()
            max_score = df[criterion_key].max()
            
            print(f"  {criterion_info['name']:20}: {mean_score:.2f} ± {std_score:.2f} (range: {min_score}-{max_score})")
    
    print()
    
    # Style analysis
    if 'style' in df.columns and 'prompt_match' in df.columns:
        print("🎨 Top Performing Styles (by Prompt Match):")
        style_scores = df.groupby('style')['prompt_match'].mean().sort_values(ascending=False)
        for style, score in style_scores.head(5).items():
            print(f"  {style:15}: {score:.2f}")
        print()
    
    # Mood analysis  
    if 'mood' in df.columns and 'creative_appeal' in df.columns:
        print("🌟 Top Performing Moods (by Creative Appeal):")
        mood_scores = df.groupby('mood')['creative_appeal'].mean().sort_values(ascending=False)
        for mood, score in mood_scores.head(5).items():
            print(f"  {mood:15}: {score:.2f}")
        print()

def generate_detailed_report(existing_evals: Dict) -> str:
    """Generate detailed evaluation report."""
    if not existing_evals:
        return "No evaluations found."
    
    df = pd.DataFrame.from_dict(existing_evals, orient='index')
    
    report_lines = [
        "# CompI Phase 1.D - Detailed Evaluation Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Images Evaluated: {len(df)}",
        "",
        "## Overall Performance Summary"
    ]
    
    # Overall statistics
    for criterion_key, criterion_info in EVALUATION_CRITERIA.items():
        if criterion_key in df.columns:
            mean_score = df[criterion_key].mean()
            std_score = df[criterion_key].std()
            report_lines.append(f"- **{criterion_info['name']}**: {mean_score:.2f} ± {std_score:.2f}")
    
    # Distribution analysis
    report_lines.extend([
        "",
        "## Score Distribution Analysis"
    ])
    
    for criterion_key, criterion_info in EVALUATION_CRITERIA.items():
        if criterion_key in df.columns:
            scores = df[criterion_key]
            report_lines.extend([
                f"",
                f"### {criterion_info['name']}",
                f"- Mean: {scores.mean():.2f}",
                f"- Median: {scores.median():.2f}",
                f"- Mode: {scores.mode().iloc[0] if not scores.mode().empty else 'N/A'}",
                f"- Range: {scores.min()}-{scores.max()}",
                f"- Distribution: " + " | ".join([f"{i}★: {(scores == i).sum()}" for i in range(1, 6)])
            ])
    
    # Style/Mood performance
    if 'style' in df.columns:
        report_lines.extend([
            "",
            "## Style Performance Analysis"
        ])
        
        for criterion_key in EVALUATION_CRITERIA.keys():
            if criterion_key in df.columns:
                style_performance = df.groupby('style')[criterion_key].agg(['mean', 'count']).sort_values('mean', ascending=False)
                report_lines.extend([
                    f"",
                    f"### {EVALUATION_CRITERIA[criterion_key]['name']} by Style",
                ])
                
                for style, (mean_score, count) in style_performance.iterrows():
                    report_lines.append(f"- {style}: {mean_score:.2f} (n={count})")
    
    # Recommendations
    report_lines.extend([
        "",
        "## Recommendations",
        "",
        "### Areas for Improvement"
    ])
    
    # Find lowest scoring criteria
    criterion_means = {}
    for criterion_key, criterion_info in EVALUATION_CRITERIA.items():
        if criterion_key in df.columns:
            criterion_means[criterion_info['name']] = df[criterion_key].mean()
    
    if criterion_means:
        lowest_criteria = sorted(criterion_means.items(), key=lambda x: x[1])[:2]
        for criterion_name, score in lowest_criteria:
            report_lines.append(f"- Focus on improving **{criterion_name}** (current: {score:.2f}/5)")
    
    report_lines.extend([
        "",
        "### Best Practices",
        "- Continue systematic evaluation for trend analysis",
        "- Experiment with parameter adjustments for low-scoring areas", 
        "- Consider A/B testing different generation approaches",
        "- Document successful style/mood combinations for reuse"
    ])
    
    return "\n".join(report_lines)

def batch_evaluate_images(images: List[Dict], scores: List[int], notes: str, output_dir: str):
    """Batch evaluate unevaluated images."""
    existing_evals = load_existing_evaluations()
    unevaluated = [img for img in images if img['filename'] not in existing_evals]
    
    if not unevaluated:
        print("✅ All images are already evaluated!")
        return
    
    print(f"📦 Batch evaluating {len(unevaluated)} images...")
    
    # Map scores to criteria
    criteria_keys = list(EVALUATION_CRITERIA.keys())
    score_dict = dict(zip(criteria_keys, scores))
    
    for i, img_data in enumerate(unevaluated):
        fname = img_data["filename"]
        img_path = os.path.join(output_dir, fname)
        
        try:
            metrics = get_image_metrics(img_path)
            save_evaluation(fname, img_data, score_dict, notes, metrics)
            print(f"  ✅ Evaluated: {fname}")
        except Exception as e:
            print(f"  ❌ Error evaluating {fname}: {e}")
    
    print(f"🎉 Batch evaluation completed!")

def list_images(images: List[Dict], existing_evals: Dict, show_evaluated: bool = True, show_unevaluated: bool = True):
    """List images with evaluation status."""
    print(f"📋 Image List ({len(images)} total)")
    print("-" * 80)
    
    for img_data in images:
        fname = img_data["filename"]
        is_evaluated = fname in existing_evals
        
        if (show_evaluated and is_evaluated) or (show_unevaluated and not is_evaluated):
            status = "✅" if is_evaluated else "❌"
            prompt = img_data.get('prompt', 'unknown')[:30]
            style = img_data.get('style', 'unknown')[:15]
            mood = img_data.get('mood', 'unknown')[:15]
            
            print(f"{status} {fname}")
            print(f"    Prompt: {prompt}... | Style: {style} | Mood: {mood}")
            
            if is_evaluated:
                eval_data = existing_evals[fname]
                scores = [f"{k}:{eval_data.get(k, 'N/A')}" for k in EVALUATION_CRITERIA.keys() if k in eval_data]
                print(f"    Scores: {' | '.join(scores[:3])}...")
            print()

def main():
    """Main CLI function."""
    args = setup_args()
    
    # Load images
    images = load_images(args.output_dir)
    if not images:
        return
    
    # Apply filters
    images = filter_images(images, args.style, args.mood)
    
    # Load existing evaluations
    existing_evals = load_existing_evaluations()
    
    # Execute commands
    if args.analyze:
        analyze_evaluations(existing_evals)
    
    elif args.report:
        report = generate_detailed_report(existing_evals)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 Report saved to: {args.output}")
        else:
            print(report)
    
    elif args.batch_score:
        batch_evaluate_images(images, args.batch_score, args.notes, args.output_dir)
    
    elif args.list_all:
        list_images(images, existing_evals, True, True)
    
    elif args.list_evaluated:
        list_images(images, existing_evals, True, False)
    
    elif args.list_unevaluated:
        list_images(images, existing_evals, False, True)
    
    else:
        print("❓ No command specified. Use --help for usage information.")

if __name__ == "__main__":
    main()

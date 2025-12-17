"""
Compare evaluation results from multiple model runs.

This script reads metrics.json files from multiple evaluation directories
and creates a comparison table.

Usage:
    python compare_evaluations.py \
        --eval_dirs ./evaluation_results/base_model ./evaluation_results/fine_tuned \
        --output comparison_table.csv
"""

import argparse
import json
from pathlib import Path
import pandas as pd


def load_metrics(eval_dir: str) -> dict:
    """Load metrics.json from evaluation directory."""
    metrics_file = Path(eval_dir) / "metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"metrics.json not found in {eval_dir}")
    
    with open(metrics_file, 'r') as f:
        return json.load(f)


def compare_evaluations(eval_dirs: list, output_file: str = None):
    """
    Compare multiple evaluation results.
    
    Args:
        eval_dirs: List of evaluation directory paths
        output_file: Optional CSV file to save comparison
    """
    results = []
    
    for eval_dir in eval_dirs:
        try:
            metrics = load_metrics(eval_dir)
            model_name = Path(eval_dir).name
            results.append({
                "model": model_name,
                "num_samples": metrics.get("num_samples", 0),
                "bleu": metrics.get("bleu", 0.0),
                "rouge1": metrics.get("rouge1", 0.0),
                "rouge2": metrics.get("rouge2", 0.0),
                "rougeL": metrics.get("rougeL", 0.0),
                "meteor": metrics.get("meteor", 0.0),
            })
        except Exception as e:
            print(f"Warning: Could not load metrics from {eval_dir}: {e}")
    
    if not results:
        print("No valid evaluation results found.")
        return
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    
    # Print comparison table
    print("\n" + "="*80)
    print("EVALUATION COMPARISON")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Save to CSV if requested
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved comparison to: {output_file}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation results from multiple models"
    )
    parser.add_argument(
        "--eval_dirs",
        nargs="+",
        required=True,
        help="List of evaluation result directories (each should contain metrics.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV file to save comparison table"
    )
    
    args = parser.parse_args()
    
    compare_evaluations(args.eval_dirs, args.output)


if __name__ == "__main__":
    main()

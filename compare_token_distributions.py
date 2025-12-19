#!/usr/bin/env python3
"""
Compare token count distributions between:
1. Original forecast text files
2. Reference texts from detailed_results.csv
3. Prediction texts from detailed_results.csv
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from collections import defaultdict

def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using the tokenizer."""
    if not text or pd.isna(text) or not str(text).strip():
        return 0
    tokens = tokenizer(str(text), add_special_tokens=False, return_tensors=None)
    return len(tokens["input_ids"])

def analyze_csv_tokens(csv_path: str, tokenizer, text_columns: list = ["reference", "prediction"]):
    """
    Analyze token counts in CSV columns.
    
    Args:
        csv_path: Path to detailed_results.csv
        tokenizer: Tokenizer instance
        text_columns: List of column names containing text to analyze
    
    Returns:
        Dictionary with token counts per column
    """
    print(f"\nLoading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Found {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    results = {}
    
    for col in text_columns:
        if col not in df.columns:
            print(f"  Warning: Column '{col}' not found in CSV. Skipping.")
            continue
        
        print(f"\n  Analyzing '{col}' column...")
        token_counts = []
        
        for idx, text in enumerate(df[col]):
            if (idx + 1) % 100 == 0:
                print(f"    Processed {idx + 1}/{len(df)} rows...")
            
            token_count = count_tokens(text, tokenizer)
            token_counts.append(token_count)
        
        results[col] = np.array(token_counts)
        print(f"    Completed: {len(token_counts)} samples")
    
    return results, df

def analyze_original_texts(text_dir: str, tokenizer, max_files: int = None):
    """
    Analyze token counts in original forecast text files.
    
    Args:
        text_dir: Directory containing date subdirectories with text files
        tokenizer: Tokenizer instance
        max_files: Maximum number of files to process (None = all)
    
    Returns:
        Array of token counts
    """
    text_dir = Path(text_dir)
    if not text_dir.exists():
        print(f"  Warning: Text directory not found: {text_dir}")
        return None
    
    print(f"\nScanning original text files in: {text_dir}")
    text_files = list(text_dir.glob("**/*.txt"))
    
    if not text_files:
        print(f"  Warning: No .txt files found in {text_dir}")
        return None
    
    print(f"  Found {len(text_files)} text files")
    
    if max_files:
        text_files = text_files[:max_files]
        print(f"  Processing first {len(text_files)} files")
    
    token_counts = []
    empty_files = 0
    errors = 0
    
    print("  Processing files...")
    for i, file_path in enumerate(text_files):
        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(text_files)} files...")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
            
            if not text:
                empty_files += 1
                continue
            
            token_count = count_tokens(text, tokenizer)
            token_counts.append(token_count)
            
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"    Error processing {file_path}: {e}")
    
    if not token_counts:
        print("  ERROR: No valid text files found!")
        return None
    
    print(f"  Completed: {len(token_counts)} files (empty: {empty_files}, errors: {errors})")
    return np.array(token_counts)

def print_statistics(name: str, token_counts: np.ndarray):
    """Print statistics for a token count distribution."""
    if token_counts is None or len(token_counts) == 0:
        print(f"\n{name}: No data available")
        return
    
    print(f"\n{'─'*70}")
    print(f"{name.upper()} - TOKEN COUNT STATISTICS")
    print(f"{'─'*70}")
    print(f"  Samples:     {len(token_counts):,}")
    print(f"  Mean:        {np.mean(token_counts):.1f} tokens")
    print(f"  Median:      {np.median(token_counts):.1f} tokens")
    print(f"  Std Dev:     {np.std(token_counts):.1f} tokens")
    print(f"  Min:         {int(np.min(token_counts))} tokens")
    print(f"  Max:         {int(np.max(token_counts))} tokens")
    
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\n  Percentiles:")
    for p in percentiles:
        val = np.percentile(token_counts, p)
        print(f"    {p:2d}th: {val:.1f} tokens")

def print_distribution(name: str, token_counts: np.ndarray):
    """Print distribution by ranges."""
    if token_counts is None or len(token_counts) == 0:
        return
    
    print(f"\n{'─'*70}")
    print(f"{name.upper()} - DISTRIBUTION BY RANGES")
    print(f"{'─'*70}")
    
    bins = [
        (0, 100, "< 100"),
        (100, 200, "100-200"),
        (200, 300, "200-300"),
        (300, 400, "300-400"),
        (400, 500, "400-500"),
        (500, 600, "500-600"),
        (600, 700, "600-700"),
        (700, 800, "700-800"),
        (800, 900, "800-900"),
        (900, 1000, "900-1000"),
        (1000, 1500, "1000-1500"),
        (1500, 2000, "1500-2000"),
        (2000, float('inf'), "> 2000"),
    ]
    
    for min_val, max_val, label in bins:
        count = np.sum((token_counts >= min_val) & (token_counts < max_val))
        pct = (count / len(token_counts)) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:15s}: {count:5d} samples ({pct:5.1f}%) {bar}")

def compare_distributions(original: np.ndarray, reference: np.ndarray, prediction: np.ndarray):
    """Compare the three distributions."""
    print(f"\n{'='*70}")
    print(f"COMPARISON ANALYSIS")
    print(f"{'='*70}")
    
    # Compare reference vs original
    if original is not None and reference is not None:
        print(f"\n{'─'*70}")
        print(f"REFERENCE vs ORIGINAL TEXT FILES")
        print(f"{'─'*70}")
        print(f"  Original mean:  {np.mean(original):.1f} tokens")
        print(f"  Reference mean: {np.mean(reference):.1f} tokens")
        diff = np.mean(reference) - np.mean(original)
        pct_diff = (diff / np.mean(original)) * 100 if np.mean(original) > 0 else 0
        print(f"  Difference:     {diff:+.1f} tokens ({pct_diff:+.1f}%)")
    
    # Compare prediction vs reference
    if reference is not None and prediction is not None:
        print(f"\n{'─'*70}")
        print(f"PREDICTION vs REFERENCE")
        print(f"{'─'*70}")
        print(f"  Reference mean:  {np.mean(reference):.1f} tokens")
        print(f"  Prediction mean: {np.mean(prediction):.1f} tokens")
        diff = np.mean(prediction) - np.mean(reference)
        pct_diff = (diff / np.mean(reference)) * 100 if np.mean(reference) > 0 else 0
        print(f"  Difference:       {diff:+.1f} tokens ({pct_diff:+.1f}%)")
        
        # Per-sample differences
        if len(reference) == len(prediction):
            per_sample_diff = prediction - reference
            print(f"\n  Per-sample differences:")
            print(f"    Mean difference:   {np.mean(per_sample_diff):+.1f} tokens")
            print(f"    Median difference: {np.median(per_sample_diff):+.1f} tokens")
            print(f"    Std Dev:           {np.std(per_sample_diff):.1f} tokens")
            
            # Count over/under predictions
            over = np.sum(per_sample_diff > 0)
            under = np.sum(per_sample_diff < 0)
            equal = np.sum(per_sample_diff == 0)
            print(f"\n    Predictions longer:  {over} samples ({over/len(per_sample_diff)*100:.1f}%)")
            print(f"    Predictions shorter: {under} samples ({under/len(per_sample_diff)*100:.1f}%)")
            print(f"    Equal length:        {equal} samples ({equal/len(per_sample_diff)*100:.1f}%)")
    
    # Compare prediction vs original
    if original is not None and prediction is not None:
        print(f"\n{'─'*70}")
        print(f"PREDICTION vs ORIGINAL TEXT FILES")
        print(f"{'─'*70}")
        print(f"  Original mean:   {np.mean(original):.1f} tokens")
        print(f"  Prediction mean: {np.mean(prediction):.1f} tokens")
        diff = np.mean(prediction) - np.mean(original)
        pct_diff = (diff / np.mean(original)) * 100 if np.mean(original) > 0 else 0
        print(f"  Difference:      {diff:+.1f} tokens ({pct_diff:+.1f}%)")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare token count distributions between original texts, references, and predictions"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to detailed_results.csv from evaluation"
    )
    parser.add_argument(
        "--text_dir",
        type=str,
        default=None,
        help="Directory containing original forecast text files (optional, for comparison)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HuggingFace model name for tokenizer"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of original text files to process (None = all)"
    )
    
    args = parser.parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Analyze CSV
    csv_results, df = analyze_csv_tokens(args.csv, tokenizer, ["reference", "prediction"])
    
    reference_tokens = csv_results.get("reference")
    prediction_tokens = csv_results.get("prediction")
    
    # Analyze original texts (optional)
    original_tokens = None
    if args.text_dir:
        original_tokens = analyze_original_texts(args.text_dir, tokenizer, args.max_files)
    
    # Print statistics
    print(f"\n{'='*70}")
    print(f"TOKEN COUNT DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")
    
    if original_tokens is not None:
        print_statistics("Original Text Files", original_tokens)
        print_distribution("Original Text Files", original_tokens)
    
    if reference_tokens is not None:
        print_statistics("Reference Texts (from CSV)", reference_tokens)
        print_distribution("Reference Texts (from CSV)", reference_tokens)
    
    if prediction_tokens is not None:
        print_statistics("Prediction Texts (from CSV)", prediction_tokens)
        print_distribution("Prediction Texts (from CSV)", prediction_tokens)
    
    # Compare distributions
    compare_distributions(original_tokens, reference_tokens, prediction_tokens)
    
    # Recommendations
    print(f"\n{'='*70}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*70}")
    
    if reference_tokens is not None:
        p95_ref = np.percentile(reference_tokens, 95)
        p99_ref = np.percentile(reference_tokens, 99)
        max_ref = np.max(reference_tokens)
        print(f"\nBased on Reference texts:")
        print(f"  Recommended max_new_tokens:")
        print(f"    Conservative: {int(p95_ref)} tokens (covers 95%)")
        print(f"    Standard:     {int(p99_ref)} tokens (covers 99%)")
        print(f"    Maximum:      {int(max_ref)} tokens (covers all)")
    
    if prediction_tokens is not None:
        p95_pred = np.percentile(prediction_tokens, 95)
        p99_pred = np.percentile(prediction_tokens, 99)
        max_pred = np.max(prediction_tokens)
        print(f"\nBased on Prediction texts:")
        print(f"  Current generation length:")
        print(f"    95th percentile: {int(p95_pred)} tokens")
        print(f"    99th percentile: {int(p99_pred)} tokens")
        print(f"    Maximum:         {int(max_pred)} tokens")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import wandb


def log_evaluation_to_wandb(
    stats_json_path: str,
    results_csv_path: str,
    project_name: str = "weather-lmm-evaluation",
    run_name: Optional[str] = None,
    run_tags: Optional[list] = None,
    config: Optional[Dict[str, Any]] = None,
    entity: Optional[str] = None,
):

    # Load statistics JSON
    print(f"Loading statistics from: {stats_json_path}")
    with open(stats_json_path, 'r') as f:
        stats_data = json.load(f)
    
    overall_stats = stats_data['overall']
    season_stats = stats_data.get('by_season', {})
    total_samples = stats_data.get('total_samples', 0)
    
    # Load detailed results CSV
    print(f"Loading detailed results from: {results_csv_path}")
    results_df = pd.read_csv(results_csv_path)
    
    # Initialize WandB run
    print(f"Initializing WandB run: {run_name}")
    wandb.init(
        project=project_name,
        name=run_name,
        tags=run_tags or [],
        config=config or {},
        entity=entity,
    )

    print("Logging overall metrics...")
    for metric in ['bleu', 'rouge1', 'rouge2', 'rougeL', 'meteor']:
        if metric in overall_stats:
            stats = overall_stats[metric]
            # These appear in the summary table for quick comparison
            wandb.summary[f"{metric}_mean"] = stats.get('mean', 0.0)
            wandb.summary[f"{metric}_median"] = stats.get('median', 0.0)
            wandb.summary[f"{metric}_std"] = stats.get('std', 0.0)
            wandb.summary[f"{metric}_min"] = stats.get('min', 0.0)
            wandb.summary[f"{metric}_max"] = stats.get('max', 0.0)
    
    wandb.summary['total_samples'] = total_samples

    print("Logging seasonal performance...")
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        if season in season_stats:
            for metric in ['bleu', 'rouge1', 'rouge2', 'rougeL', 'meteor']:
                if metric in season_stats[season]:
                    mean_val = season_stats[season][metric].get('mean', 0.0)
                    wandb.log({f"season/{season}/{metric}": mean_val})
    
    print("Logging metric distributions...")
    for metric in ['bleu', 'rouge1', 'rouge2', 'rougeL', 'meteor']:
        if metric in results_df.columns:
            values = results_df[metric].dropna().values
            if len(values) > 0:
                wandb.log({
                    f"distribution/{metric}": wandb.Histogram(values)
                })

    print("Logging detailed results table...")
    # Limit to first 100 rows to avoid WandB limits
    table_df = results_df.head(100)[['sample_id', 'bleu', 'rouge1', 'rouge2', 'rougeL', 'meteor']]
    wandb.log({"detailed_results": wandb.Table(dataframe=table_df)})

    print("Creating custom charts...")
    
    # Chart 1: BLEU vs ROUGE-1 scatter plot
    if 'bleu' in results_df.columns and 'rouge1' in results_df.columns:
        data = [[x, y] for x, y in zip(results_df['bleu'].head(100), results_df['rouge1'].head(100))]
        table = wandb.Table(data=data, columns=["BLEU", "ROUGE-1"])
        wandb.log({
            "charts/bleu_vs_rouge1": wandb.plot.scatter(table, "BLEU", "ROUGE-1", title="BLEU vs ROUGE-1")
        })
    
    # Chart 2: Metric comparison bar chart
    metrics_data = []
    for metric in ['bleu', 'rouge1', 'rouge2', 'rougeL', 'meteor']:
        if metric in overall_stats:
            metrics_data.append([metric.upper(), overall_stats[metric].get('mean', 0.0)])
    
    if metrics_data:
        table = wandb.Table(data=metrics_data, columns=["Metric", "Mean Score"])
        wandb.log({
            "charts/metric_comparison": wandb.plot.bar(table, "Metric", "Mean Score", title="Overall Metrics")
        })
    
    print(f"✓ Successfully logged to WandB: {wandb.run.url}")
    wandb.finish()


def parse_config_string(config_str: str) -> Dict[str, Any]:
    config = {}
    if not config_str:
        return config
    
    for item in config_str.split():
        if '=' in item:
            key, value = item.split('=', 1)
            # Try to parse as number
            try:
                if '.' in value or 'e' in value.lower():
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep as string
            config[key] = value
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Log evaluation results to Weights & Biases"
    )
    parser.add_argument(
        "--stats_json",
        type=str,
        required=True,
        help="Path to statistics.json from visualize_evaluation.py"
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        required=True,
        help="Path to detailed_results.csv from evaluate.py"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="weather-lmm-evaluation",
        help="WandB project name"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name for this run (e.g., 'baseline', 'lora-r64-lr1e4')"
    )
    parser.add_argument(
        "--run_tags",
        type=str,
        nargs='+',
        default=None,
        help="Tags for this run (e.g., 'baseline pretrained' or 'lora finetuned')"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config as space-separated key=value pairs (e.g., 'lora_r=64 learning_rate=1e-4')"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="WandB username or team name (optional)"
    )
    
    args = parser.parse_args()
    
    # Parse config string
    config = parse_config_string(args.config) if args.config else None
    
    # Log to WandB
    log_evaluation_to_wandb(
        stats_json_path=args.stats_json,
        results_csv_path=args.results_csv,
        project_name=args.project,
        run_name=args.run_name,
        run_tags=args.run_tags,
        config=config,
        entity=args.entity,
    )


if __name__ == "__main__":
    main()
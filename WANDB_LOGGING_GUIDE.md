# WandB Logging Guide for WeatherLMM

Log evaluation results to Weights & Biases for easy side-by-side model comparison.

---

## Prerequisites

```bash
#everything below as generated with Claude.
# Install WandB
pip install wandb

# Login (one-time setup)
wandb login
```

Paste your API key from https://wandb.ai/authorize when prompted.

---

## Quick Start

### 1. Run Evaluation & Visualization

```bash
# Evaluate model
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/weather_lora \
    --output_dir ./lora_eval

# Create HTML visualization (generates statistics.json)
python visualize_evaluation.py \
    --results_csv ./lora_eval/detailed_results.csv \
    --test_csv ./manifests/manifest_test.csv \
    --output_dir ./lora_eval
```

### 2. Log to WandB

```bash
python Logging_visualize_to_wandb.py \
    --stats_json ./lora_eval/statistics.json \
    --results_csv ./lora_eval/detailed_results.csv \
    --run_name "lora-r64-3epochs" \
    --run_tags lora finetuned \
    --config lora_r=64 learning_rate=1e-4 epochs=3
```

View your results at the URL printed in the output!

---

## Compare Baseline vs Fine-Tuned

### Evaluate Baseline

```bash
# Evaluate
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --output_dir ./baseline_eval

# Visualize
python visualize_evaluation.py \
    --results_csv ./baseline_eval/detailed_results.csv \
    --test_csv ./manifests/manifest_test.csv \
    --output_dir ./baseline_eval

# Log to WandB
python Logging_visualize_to_wandb.py \
    --stats_json ./baseline_eval/statistics.json \
    --results_csv ./baseline_eval/detailed_results.csv \
    --run_name "baseline-pretrained" \
    --run_tags baseline
```

### Evaluate Fine-Tuned

```bash
# Evaluate
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/weather_lora \
    --output_dir ./lora_eval

# Visualize
python visualize_evaluation.py \
    --results_csv ./lora_eval/detailed_results.csv \
    --test_csv ./manifests/manifest_test.csv \
    --output_dir ./lora_eval

# Log to WandB
python Logging_visualize_to_wandb.py \
    --stats_json ./lora_eval/statistics.json \
    --results_csv ./lora_eval/detailed_results.csv \
    --run_name "lora-r64-3epochs" \
    --run_tags lora \
    --config lora_r=64 epochs=3 learning_rate=1e-4
```

### View Comparison

Go to https://wandb.ai/yourname/weather-lmm-evaluation

WandB shows both runs side-by-side with automatic comparison!

---

## Command-Line Arguments

| Argument | Required | Description | Example |
|----------|----------|-------------|---------|
| `--stats_json` | Yes | Path to statistics.json | `./lora_eval/statistics.json` |
| `--results_csv` | Yes | Path to detailed_results.csv | `./lora_eval/detailed_results.csv` |
| `--run_name` | Yes | Name for this run | `baseline` or `lora-r64-lr1e4` |
| `--run_tags` | No | Tags (space-separated) | `baseline pretrained` or `lora finetuned` |
| `--config` | No | Config (space-separated key=value) | `lora_r=64 learning_rate=1e-4` |
| `--project` | No | WandB project name | `weather-lmm-evaluation` (default) |
| `--entity` | No | WandB username/team | Your username |

---

## What Gets Logged

### Summary Metrics
- Mean, median, std, min, max for BLEU, ROUGE, METEOR
- Appears in WandB comparison table

### Seasonal Performance
- Performance breakdown by Winter/Spring/Summer/Fall
- Shows if model performs better in certain seasons

### Distributions
- Histograms of score distributions
- Shows consistency vs variance

### Charts
- BLEU vs ROUGE-1 scatter plot
- Metric comparison bar chart

---

## Example Use Cases

### Compare Different LoRA Ranks

```bash
# Train & evaluate with r=16, r=32, r=64
# Then log each:

python Logging_visualize_to_wandb.py \
    --stats_json ./lora_r16_eval/statistics.json \
    --results_csv ./lora_r16_eval/detailed_results.csv \
    --run_name "lora-r16" \
    --config lora_r=16

python Logging_visualize_to_wandb.py \
    --stats_json ./lora_r32_eval/statistics.json \
    --results_csv ./lora_r32_eval/detailed_results.csv \
    --run_name "lora-r32" \
    --config lora_r=32

python Logging_visualize_to_wandb.py \
    --stats_json ./lora_r64_eval/statistics.json \
    --results_csv ./lora_r64_eval/detailed_results.csv \
    --run_name "lora-r64" \
    --config lora_r=64
```

WandB automatically ranks them by performance!

### Track Training Checkpoints

```bash
# Log each epoch checkpoint
python Logging_visualize_to_wandb.py \
    --stats_json ./epoch1_eval/statistics.json \
    --results_csv ./epoch1_eval/detailed_results.csv \
    --run_name "lora-epoch1" \
    --run_tags progression \
    --config epoch=1

python Logging_visualize_to_wandb.py \
    --stats_json ./epoch2_eval/statistics.json \
    --results_csv ./epoch2_eval/detailed_results.csv \
    --run_name "lora-epoch2" \
    --run_tags progression \
    --config epoch=2

python Logging_visualize_to_wandb.py \
    --stats_json ./epoch3_eval/statistics.json \
    --results_csv ./epoch3_eval/detailed_results.csv \
    --run_name "lora-epoch3" \
    --run_tags progression \
    --config epoch=3
```

---

## Best Practices

### 1. Use Descriptive Names
✅ Good: `baseline-pretrained`, `lora-r64-lr1e4-3epochs`
❌ Bad: `run1`, `test`, `model_v2`

### 2. Tag Consistently
- Model type: `baseline`, `lora`
- Status: `pretrained`, `finetuned`
- Purpose: `comparison`, `progression`, `final`

### 3. Log Hyperparameters
Always include relevant config:
```bash
--config lora_r=64 learning_rate=1e-4 epochs=3 batch_size=1
```

---

## Troubleshooting

### WandB Login Issues
```bash
# Re-login
wandb logout
wandb login

# Or set API key directly
export WANDB_API_KEY="your-api-key-here"
```

### Offline Mode (HPC/No Internet)
```bash
# Run in offline mode
export WANDB_MODE=offline
python Logging_visualize_to_wandb.py ...

# Later, sync when online
wandb sync ./wandb/offline-run-xxxxx
```

### Missing statistics.json
Make sure you ran `visualize_evaluation.py` first - it generates `statistics.json`.

---

## When to Use WandB vs HTML

### Use HTML Visualization
- Single final model
- Reports for papers/presentations
- Detailed image + text comparisons
- Offline, self-contained reports

### Use WandB
- Comparing multiple models
- Tracking experiments over time
- Team collaboration
- Quick hyperparameter comparison

### Use Both (Recommended!)
HTML for detailed analysis, WandB for comparison.

---

## Full Pipeline Example

```bash
# 1. Train model
python LoRA_Training.py \
    --train_csv ./manifests/manifest_train.csv \
    --output_dir ./checkpoints/lora

# 2. Evaluate
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/lora \
    --output_dir ./eval

# 3. Visualize (generates statistics.json)
python visualize_evaluation.py \
    --results_csv ./eval/detailed_results.csv \
    --test_csv ./manifests/manifest_test.csv \
    --output_dir ./eval

# 4. Log to WandB
python Logging_visualize_to_wandb.py \
    --stats_json ./eval/statistics.json \
    --results_csv ./eval/detailed_results.csv \
    --run_name "lora-r64-3epochs" \
    --run_tags lora \
    --config lora_r=64 epochs=3
```

---

## Quick Reference

```bash
# Log baseline
python Logging_visualize_to_wandb.py \
    --stats_json ./baseline_eval/statistics.json \
    --results_csv ./baseline_eval/detailed_results.csv \
    --run_name "baseline"

# Log fine-tuned
python Logging_visualize_to_wandb.py \
    --stats_json ./lora_eval/statistics.json \
    --results_csv ./lora_eval/detailed_results.csv \
    --run_name "lora-r64" \
    --run_tags lora \
    --config lora_r=64 learning_rate=1e-4 epochs=3

# View dashboard
# URL printed after logging, or go to:
# https://wandb.ai/yourname/weather-lmm-evaluation
```

---

## Resources

- **WandB Docs**: https://docs.wandb.ai/
- **Python API**: https://docs.wandb.ai/ref/python

# WeatherLMM Evaluation Guide

Complete guide for evaluating fine-tuned WeatherLMM models on test sets.

---

## Overview

The evaluation script (`evaluate.py`) provides comprehensive evaluation of your fine-tuned models by:
1. **Loading models** (base or LoRA fine-tuned)
2. **Running inference** on test set
3. **Computing metrics** (BLEU, ROUGE, METEOR)
4. **Generating reports** with detailed comparisons

---

## Prerequisites

### Install Evaluation Dependencies

```bash
# Core evaluation libraries
pip install nltk rouge-score tqdm

# If not already installed
pip install pandas torch transformers peft
```

### Download NLTK Data

The script will automatically download required NLTK data, but you can also do it manually:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

---

## Quick Start

### Evaluate Fine-Tuned LoRA Model (Greedy Decoding)

```bash
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/weather_lora \
    --output_dir ./evaluation_results \
    --max_samples 100
```

### Evaluate with Sampling (More Diverse Outputs)

```bash
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/weather_lora \
    --output_dir ./evaluation_results \
    --do_sample \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_samples 100
```

### Evaluate Base Model (No Fine-Tuning)

```bash
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_name Qwen/Qwen2.5-VL-3B-Instruct \
    --output_dir ./evaluation_results/base_model \
    --max_samples 100
```

---

## Command-Line Options

```bash
python evaluate.py \
    --test_csv PATH_TO_TEST_CSV \          # Required: Test manifest CSV
    --model_name MODEL_NAME \              # Base model (if --model_path not provided)
    --model_path PATH_TO_CHECKPOINT \      # Path to LoRA checkpoint directory
    --output_dir ./evaluation_results \     # Output directory
    --batch_size 1 \                        # Batch size (currently only 1)
    --max_samples 100 \                    # Limit samples for quick testing
    --max_new_tokens 2048 \                 # Max tokens to generate
    --device cuda \                         # Device (cuda or cpu)
    --do_sample \                           # Enable sampling (optional, default: greedy)
    --temperature 0.7 \                     # Sampling temperature (only with --do_sample)
    --top_p 0.9                            # Nucleus sampling (only with --do_sample)
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--test_csv` | Yes | - | Path to test manifest CSV |
| `--model_name` | No* | `Qwen/Qwen2.5-VL-3B-Instruct` | Base model name |
| `--model_path` | No* | `None` | Path to LoRA checkpoint |
| `--output_dir` | No | `./evaluation_results` | Output directory |
| `--batch_size` | No | `1` | Batch size (currently only 1) |
| `--max_samples` | No | `None` | Limit number of samples |
| `--max_new_tokens` | No | `2048` | Max generation length |
| `--device` | No | `cuda` | Device to use |
| `--do_sample` | No | `False` | Enable sampling instead of greedy decoding |
| `--temperature` | No | `0.7` | Sampling temperature (only used with `--do_sample`) |
| `--top_p` | No | `0.9` | Nucleus sampling parameter (only used with `--do_sample`) |

*Either `--model_name` (for base model) or `--model_path` (for fine-tuned) must be provided.

---

## Decoding Strategies

The evaluation script supports two decoding strategies:

### Greedy Decoding (Default)

**Deterministic**: Always selects the most likely token at each step.

- **Use when**: You want reproducible, consistent results
- **Best for**: Comparing models fairly, debugging, production use
- **Behavior**: Same input → same output every time

```bash
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/weather_lora \
    --output_dir ./evaluation_results
    # No --do_sample flag = greedy decoding
```

### Sampling

**Stochastic**: Samples from the probability distribution over tokens.

- **Use when**: You want diverse outputs, testing model creativity
- **Best for**: Exploring model behavior, generating multiple variants
- **Behavior**: Same input → different outputs each run

**Parameters:**
- `--temperature`: Controls randomness (0.1-2.0)
  - Lower (0.1-0.5): More conservative, focused outputs
  - Medium (0.7-1.0): Balanced creativity and coherence
  - Higher (1.2-2.0): More diverse, potentially less coherent
- `--top_p`: Nucleus sampling threshold (0.0-1.0)
  - Filters tokens to top-p probability mass
  - Lower (0.5-0.8): More focused
  - Higher (0.9-0.95): More diverse

```bash
# Balanced sampling (recommended)
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/weather_lora \
    --do_sample \
    --temperature 0.7 \
    --top_p 0.9

# More conservative sampling
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/weather_lora \
    --do_sample \
    --temperature 0.5 \
    --top_p 0.8

# More creative sampling
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/weather_lora \
    --do_sample \
    --temperature 1.2 \
    --top_p 0.95
```

**Note**: If your model produces identical outputs with greedy decoding, try sampling to see if it generates different forecasts for different inputs. This helps diagnose whether the issue is deterministic decoding or model behavior.

---

## Evaluation Metrics

The script computes the following metrics:

### BLEU Score
- Measures n-gram precision between reference and prediction
- Range: 0.0 to 1.0 (higher is better)
- Good for: Overall text similarity

### ROUGE Scores
- **ROUGE-1**: Unigram overlap (recall)
- **ROUGE-2**: Bigram overlap (recall)
- **ROUGE-L**: Longest common subsequence (LCS-based)
- Range: 0.0 to 1.0 (higher is better)
- Good for: Content coverage and fluency

### METEOR Score
- Considers synonyms and word order
- Range: 0.0 to 1.0 (higher is better)
- Good for: Semantic similarity

---

## Output Files

After evaluation, the following files are created in `--output_dir`:

### 1. `metrics.json`
Summary metrics for the entire test set:
```json
{
  "num_samples": 100,
  "bleu": 0.2345,
  "rouge1": 0.4567,
  "rouge2": 0.3456,
  "rougeL": 0.4321,
  "meteor": 0.3789
}
```

### 2. `detailed_results.csv`
Per-sample metrics with predictions and references:
```csv
sample_id,prediction,reference,bleu,rouge1,rouge2,rougeL,meteor
0,"Generated forecast...","Reference forecast...",0.23,0.45,0.34,0.43,0.38
...
```

### 3. `predictions_vs_references.csv`
Side-by-side comparison for manual inspection:
```csv
sample_id,reference,prediction,bleu,rouge1,rouge2,rougeL,meteor
0,"Reference text...","Predicted text...",0.23,0.45,0.34,0.43,0.38
...
```

---

## Complete Evaluation Workflow

### Step 1: Prepare Test Set

Ensure you have a test manifest (created by `create_full_image_manifest.py`):

```bash
# Create manifests with train/test split
python create_full_image_manifest.py \
    --text_dir ./forecast_text_Cleaned \
    --image_base_dir ./images \
    --output_dir ./manifests \
    --test_year 2025
```

This creates `./manifests/manifest_test.csv`.

### Step 2: Evaluate Base Model (Baseline)

```bash
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_name Qwen/Qwen2.5-VL-3B-Instruct \
    --output_dir ./evaluation_results/base_model \
    --max_samples 100
```

### Step 3: Train Your Model

```bash
python LoRA_Training.py \
    --train_csv ./manifests/manifest_train.csv \
    --eval_csv ./manifests/manifest_test.csv \
    --output_dir ./checkpoints/weather_lora \
    --num_train_epochs 3
```

### Step 4: Evaluate Fine-Tuned Model

```bash
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/weather_lora \
    --output_dir ./evaluation_results/fine_tuned \
    --max_samples 100
```

### Step 5: Compare Results

Compare metrics from `base_model/metrics.json` and `fine_tuned/metrics.json`:

```bash
# View base model metrics
cat ./evaluation_results/base_model/metrics.json

# View fine-tuned metrics
cat ./evaluation_results/fine_tuned/metrics.json
```

---

## Comparing Multiple Models

To compare multiple checkpoints or configurations:

```bash
# Evaluate checkpoint 1
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/weather_lora_epoch1 \
    --output_dir ./evaluation_results/epoch1

# Evaluate checkpoint 2
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/weather_lora_epoch2 \
    --output_dir ./evaluation_results/epoch2

# Evaluate checkpoint 3
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/weather_lora_epoch3 \
    --output_dir ./evaluation_results/epoch3
```

Then compare the `metrics.json` files from each directory.

---

## Tips for Evaluation

### 1. Start with Small Sample
For quick testing, use `--max_samples`:

```bash
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/weather_lora \
    --max_samples 10  # Quick test with 10 samples
```

### 2. Full Evaluation
Remove `--max_samples` for complete evaluation:

```bash
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/weather_lora
    # Evaluates all samples in test set
```

### 3. CPU Evaluation
If GPU is unavailable:

```bash
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/weather_lora \
    --device cpu
```

Note: CPU evaluation will be significantly slower.

### 4. Inspect Individual Predictions
Open `predictions_vs_references.csv` in a spreadsheet or text editor to:
- Identify patterns in errors
- Find best/worst examples
- Understand model behavior

### 5. Monitor Generation Length
Adjust `--max_new_tokens` if predictions are too short or long:

```bash
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/weather_lora \
    --max_new_tokens 1024  # Shorter forecasts
```

### 6. Try Sampling for Diverse Outputs
If you're getting identical predictions, try sampling to see if the model produces different forecasts:

```bash
# Test with sampling
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/weather_lora \
    --do_sample \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_samples 10  # Quick test
```

This helps diagnose whether identical outputs are due to:
- **Deterministic decoding** (fixed by using `--do_sample`)
- **Model behavior** (model always generates same template regardless of input)

---

## Troubleshooting

### Import Errors

**NLTK not found:**
```bash
pip install nltk
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

**rouge-score not found:**
```bash
pip install rouge-score
```

**PEFT not found (for LoRA models):**
```bash
pip install peft
```

### Image Not Found Errors

- Verify image paths in test manifest CSV are correct
- Check that images exist at specified paths
- Ensure `--image_base_dir` matches your actual directory structure

### Out of Memory

- Reduce `--max_new_tokens`
- Use smaller model (`Qwen/Qwen2.5-VL-3B-Instruct` instead of 7B)
- Evaluate in smaller batches (currently only batch_size=1 supported)

### Slow Evaluation

- Use GPU (`--device cuda`)
- Reduce `--max_samples` for quick tests
- Use FP16/BF16 if supported (handled automatically)

---

## Interpreting Results

### Good Metrics
- **BLEU > 0.2**: Reasonable similarity
- **ROUGE-1 > 0.4**: Good content coverage
- **ROUGE-L > 0.3**: Decent fluency
- **METEOR > 0.3**: Good semantic similarity

### Comparing Models
- **Higher is better** for all metrics
- Compare fine-tuned vs base model
- Look for consistent improvements across metrics
- Check individual samples in `predictions_vs_references.csv`

### Common Issues

**Low BLEU but high ROUGE-1:**
- Model generates relevant content but different wording
- May indicate good semantic understanding but different style

**High BLEU but low ROUGE-L:**
- Model matches n-grams but poor sentence structure
- May need more training or better prompt

**All metrics low:**
- Model may not be learning effectively
- Check training data quality
- Verify model loaded correctly

---

## Advanced Usage

### Custom Evaluation

You can modify `evaluate.py` to:
- Add custom metrics
- Change generation parameters
- Implement domain-specific evaluation
- Add visualization of results

### Batch Evaluation Script

Create a script to evaluate multiple models:

```bash
#!/bin/bash
# evaluate_all.sh

MODELS=(
    "./checkpoints/weather_lora_epoch1"
    "./checkpoints/weather_lora_epoch2"
    "./checkpoints/weather_lora_epoch3"
)

for model in "${MODELS[@]}"; do
    echo "Evaluating $model..."
    python evaluate.py \
        --test_csv ./manifests/manifest_test.csv \
        --model_path "$model" \
        --output_dir "./evaluation_results/$(basename $model)"
done
```

---

## Visualization

### Create Interactive HTML Report

After running evaluation, create a visual report with images, text comparisons, and statistics:

```bash
python visualize_evaluation.py \
    --results_csv ./evaluation_results/detailed_results.csv \
    --test_csv ./manifests/manifest_test.csv \
    --output_dir ./evaluation_visualization \
    --model_name "WeatherLMM Fine-Tuned"
```

This creates:
- **`evaluation_report.html`**: Interactive HTML report with:
  - Original weather chart images (12 per sample)
  - Side-by-side text comparison (reference vs prediction)
  - Full statistical breakdown (mean, median, std, min, max, quartiles)
  - Season-by-season analysis (Winter, Spring, Summer, Fall)
  - Per-sample metrics and visualizations
- **`statistics.json`**: Machine-readable statistics

### Viewing the Report

Open `evaluation_report.html` in your web browser:
```bash
# On macOS
open ./evaluation_visualization/evaluation_report.html

# On Linux
xdg-open ./evaluation_visualization/evaluation_report.html

# On Windows
start ./evaluation_visualization/evaluation_report.html
```

The report includes:
- **Overview Tab**: Overall statistics and metric distributions
- **Samples Tab**: First 20 samples with images and text comparisons
- **Season Analysis Tab**: Performance breakdown by season

### Visualization Features

- **Image Display**: All 12 weather charts per sample
- **Text Comparison**: Reference and prediction side-by-side
- **Metric Badges**: Color-coded performance indicators
- **Statistical Analysis**: Comprehensive statistics for each metric
- **Season Breakdown**: Performance analysis by season

---

## Next Steps

After evaluation:

1. **Visualize results**: Create HTML report with `visualize_evaluation.py`
2. **Analyze results**: Review `predictions_vs_references.csv` for patterns
3. **Identify improvements**: Find common error types
4. **Iterate**: Adjust training or model configuration
5. **Compare**: Benchmark against baselines and other models
6. **Report**: Document findings and metrics

---

## Quick Reference

### Evaluate Fine-Tuned Model (Greedy Decoding)
```bash
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/weather_lora \
    --output_dir ./evaluation_results
```

### Evaluate with Sampling
```bash
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/weather_lora \
    --output_dir ./evaluation_results \
    --do_sample \
    --temperature 0.7 \
    --top_p 0.9
```

### Evaluate Base Model
```bash
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_name Qwen/Qwen2.5-VL-3B-Instruct \
    --output_dir ./evaluation_results/base_model
```

### Quick Test (10 samples)
```bash
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/weather_lora \
    --max_samples 10
```

### Visualize Results
```bash
python visualize_evaluation.py \
    --results_csv ./evaluation_results/detailed_results.csv \
    --test_csv ./manifests/manifest_test.csv \
    --output_dir ./evaluation_visualization
```

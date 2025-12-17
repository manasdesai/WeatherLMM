# Evaluation Visualization Guide

Complete guide for visualizing evaluation results with interactive HTML reports.

---

## Overview

The visualization script (`visualize_evaluation.py`) creates an interactive HTML report showing:
- **Original images**: All 12 weather chart images per sample
- **Text comparisons**: Reference vs predicted text side-by-side
- **Full statistics**: Mean, median, std, min, max, quartiles for all metrics
- **Season analysis**: Performance breakdown by Winter, Spring, Summer, Fall

---

## Quick Start

### Step 1: Run Evaluation

First, evaluate your model:

```bash
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/weather_lora \
    --output_dir ./evaluation_results
```

This creates `detailed_results.csv` with all predictions and metrics.

### Step 2: Create Visualization

Generate the HTML report:

```bash
python visualize_evaluation.py \
    --results_csv ./evaluation_results/detailed_results.csv \
    --test_csv ./manifests/manifest_test.csv \
    --output_dir ./evaluation_visualization \
    --model_name "WeatherLMM Fine-Tuned"
```

### Step 3: View Report

Open the HTML file in your browser:

```bash
# macOS
open ./evaluation_visualization/evaluation_report.html

# Linux
xdg-open ./evaluation_visualization/evaluation_report.html

# Windows
start ./evaluation_visualization/evaluation_report.html
```

---

## Command-Line Options

```bash
python visualize_evaluation.py \
    --results_csv PATH_TO_DETAILED_RESULTS \    # Required: detailed_results.csv from evaluate.py
    --test_csv PATH_TO_TEST_MANIFEST \          # Required: Test manifest CSV (for image paths)
    --output_dir ./evaluation_visualization \   # Output directory
    --model_name "Model Name"                    # Model name for report title
```

---

## Report Features

### 1. Overview Tab

**Overall Statistics Cards:**
- Visual cards for each metric (BLEU, ROUGE-1/2/L, METEOR)
- Mean value prominently displayed
- Median, std dev, min, max, quartiles shown

**Detailed Statistics Table:**
- Complete breakdown of all metrics
- Easy to compare across metrics
- Includes quartiles for distribution analysis

### 2. Samples Tab

**Per-Sample Display:**
- **Sample Header**: Sample ID and metric badges (color-coded)
- **Weather Charts**: All 12 images displayed in a grid
  - Temperature and geopotential at pressure levels
  - 2m temperature and wind
  - Thickness and mean sea level pressure
  - Wind and relative humidity at pressure levels
- **Text Comparison**: Side-by-side view
  - **Reference**: Ground truth forecast (green box)
  - **Prediction**: Model output (blue box)

**Metric Badges:**
- Color-coded by performance:
  - 🟢 Green: Good performance
  - 🔴 Red: Poor performance

### 3. Season Analysis Tab

**Season-by-Season Breakdown:**
- Separate statistics for each season:
  - **Winter** (Dec, Jan, Feb)
  - **Spring** (Mar, Apr, May)
  - **Summer** (Jun, Jul, Aug)
  - **Fall** (Sep, Oct, Nov)
- Sample count per season
- All metrics broken down by season

---

## Output Files

### `evaluation_report.html`
Interactive HTML report with:
- Tabbed interface for easy navigation
- Responsive design (works on desktop and tablet)
- Embedded images (base64 encoded, no external dependencies)
- Color-coded metrics and statistics

### `statistics.json`
Machine-readable statistics:
```json
{
  "overall": {
    "bleu": {"mean": 0.2345, "median": 0.2100, ...},
    "rouge1": {...},
    ...
  },
  "by_season": {
    "Winter": {"bleu": {...}, ...},
    "Spring": {...},
    ...
  },
  "total_samples": 100
}
```

---

## Use Cases

### 1. Model Comparison

Compare multiple models:

```bash
# Evaluate model 1
python evaluate.py --model_path ./checkpoints/model1 --output_dir ./results1
python visualize_evaluation.py --results_csv ./results1/detailed_results.csv --test_csv ./manifests/manifest_test.csv --output_dir ./viz1 --model_name "Model 1"

# Evaluate model 2
python evaluate.py --model_path ./checkpoints/model2 --output_dir ./results2
python visualize_evaluation.py --results_csv ./results2/detailed_results.csv --test_csv ./manifests/manifest_test.csv --output_dir ./viz2 --model_name "Model 2"
```

Then open both HTML files side-by-side to compare.

### 2. Error Analysis

1. Open the HTML report
2. Go to "Samples" tab
3. Look for samples with low metric scores (red badges)
4. Compare reference vs prediction to identify error patterns
5. Check if errors are season-specific (use Season Analysis tab)

### 3. Season-Specific Performance

1. Open the HTML report
2. Go to "Season Analysis" tab
3. Compare metrics across seasons
4. Identify which seasons the model performs best/worst on
5. Use this to guide training data augmentation

### 4. Statistical Analysis

1. Open the HTML report
2. Go to "Overview" tab
3. Review the detailed statistics table
4. Check quartiles to understand distribution
5. Compare std dev to assess consistency

---

## Tips

### 1. Large Datasets

For large test sets, the visualization shows the first 20 samples. To see more:
- Open `detailed_results.csv` directly
- Filter by specific metrics or seasons
- Create custom visualizations using the data

### 2. Image Loading

Images are embedded as base64 in the HTML, so:
- Report file may be large (especially with many samples)
- All images load automatically (no external dependencies)
- Works offline once generated

### 3. Customization

To customize the visualization:
- Edit `visualize_evaluation.py`
- Modify HTML template in `generate_html()` function
- Adjust CSS styles for different appearance
- Add additional metrics or visualizations

### 4. Sharing Reports

The HTML report is self-contained:
- Can be shared via email or file sharing
- No server needed (just open HTML file)
- Works on any modern browser
- All images embedded (no broken links)

---

## Troubleshooting

### Images Not Displaying

- Verify image paths in test manifest CSV are correct
- Check that images exist at specified paths
- Ensure images are readable (permissions)

### Season Not Detected

- Check image filename format: `{variable}_{init_time}_{lead_time}_{date}.1.png`
- Date should be in YYYYMMDD format
- If dates aren't detected, season will be "None"

### Report Too Large

- Reduce number of samples in evaluation (`--max_samples`)
- Or modify script to show fewer samples in HTML
- Consider generating separate reports per season

### Statistics Missing

- Ensure `detailed_results.csv` contains all required columns
- Check that metrics were computed correctly in evaluation
- Verify no NaN values in metrics columns

---

## Example Workflow

```bash
# 1. Evaluate model
python evaluate.py \
    --test_csv ./manifests/manifest_test.csv \
    --model_path ./checkpoints/weather_lora \
    --output_dir ./evaluation_results

# 2. Create visualization
python visualize_evaluation.py \
    --results_csv ./evaluation_results/detailed_results.csv \
    --test_csv ./manifests/manifest_test.csv \
    --output_dir ./evaluation_visualization \
    --model_name "WeatherLMM v1.0"

# 3. View report
open ./evaluation_visualization/evaluation_report.html

# 4. Analyze results
# - Check Overview tab for overall performance
# - Review Samples tab for specific examples
# - Use Season Analysis to identify seasonal patterns
```

---

## Advanced Usage

### Custom Statistics

Modify `compute_statistics()` function to add:
- Percentiles (90th, 95th, 99th)
- Confidence intervals
- Skewness and kurtosis
- Custom domain-specific metrics

### Additional Visualizations

Add to HTML template:
- Histograms of metric distributions
- Box plots for season comparison
- Scatter plots (metric vs metric)
- Time series (if date information available)

### Export Options

Extend script to export:
- PDF reports (using weasyprint or similar)
- LaTeX tables for papers
- CSV summaries
- JSON for programmatic access

---

## Quick Reference

### Create Visualization
```bash
python visualize_evaluation.py \
    --results_csv ./evaluation_results/detailed_results.csv \
    --test_csv ./manifests/manifest_test.csv \
    --output_dir ./evaluation_visualization
```

### View Report
```bash
open ./evaluation_visualization/evaluation_report.html
```

### Complete Workflow
```bash
# Evaluate
python evaluate.py --test_csv ./manifests/manifest_test.csv --model_path ./checkpoints/weather_lora --output_dir ./results

# Visualize
python visualize_evaluation.py --results_csv ./results/detailed_results.csv --test_csv ./manifests/manifest_test.csv --output_dir ./viz

# View
open ./viz/evaluation_report.html
```

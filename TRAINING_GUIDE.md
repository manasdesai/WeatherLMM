# WeatherLMM LoRA Training Guide

Complete guide for training the WeatherLMM model using LoRA fine-tuning with HuggingFace Trainer.

---

## Overview

This guide covers the complete workflow for LoRA fine-tuning of Qwen2.5-VL on weather forecast data:
1. **Create training manifests** from your data
2. **Train/test split** (automatic)
3. **LoRA fine-tuning** using HuggingFace Trainer

The training script (`LoRA_Training.py`) handles 12 weather chart images per example and generates text forecasts.

---

## Prerequisites

### 1. Install Dependencies

```bash
# Install core requirements
pip install -r requirements.txt

# Install LoRA training dependencies
pip install peft
pip install pandas  # For CSV handling

# Optional: For 4-bit quantization (QLoRA) - requires GPU
pip install bitsandbytes
```

### 2. Data Structure

Your data should be organized as:
```
project/
├── forecast_text_Cleaned/          # Text forecast files
│   ├── 2020-01-01/
│   │   ├── 03-55.txt
│   │   ├── 19-31.txt
│   │   └── ...
│   ├── 2020-01-02/
│   └── ...
└── images/                          # Weather chart images (12 per forecast)
    ├── t_z_1000_0000_12_20200101.1.png
    ├── t_z_200_0000_12_20200101.1.png
    ├── t_z_500_0000_12_20200101.1.png
    ├── t_z_700_0000_12_20200101.1.png
    ├── t_z_850_0000_12_20200101.1.png
    ├── t2m_wind10m_0000_12_20200101.1.png
    ├── thickness_mslp_0000_12_20200101.1.png
    ├── uv_rh_1000_0000_12_20200101.1.png
    ├── uv_rh_200_0000_12_20200101.1.png
    ├── uv_rh_500_0000_12_20200101.1.png
    ├── uv_rh_700_0000_12_20200101.1.png
    └── uv_rh_850_0000_12_20200101.1.png
```

**12 Image Types** (per forecast):
- `t_z_1000`, `t_z_200`, `t_z_500`, `t_z_700`, `t_z_850` - Temperature and geopotential at pressure levels
- `t2m_wind10m` - 2m temperature and 10m wind
- `thickness_mslp` - 500-1000 hPa thickness and mean sea level pressure
- `uv_rh_1000`, `uv_rh_200`, `uv_rh_500`, `uv_rh_700`, `uv_rh_850` - Wind and relative humidity at pressure levels

---

## Step 1: Create Training Manifests

Use `create_full_image_manifest.py` to generate training manifests from your data.

### Basic Usage

```bash
python create_full_image_manifest.py \
    --text_dir ./forecast_text_Cleaned \
    --image_base_dir ./images \
    --output_dir ./manifests \
    --start_date 2016-01-01 \
    --end_date 2025-12-12
```

### Command Options

```bash
python create_full_image_manifest.py \
    --text_dir PATH_TO_TEXT_DIR \           # Required: Directory with YYYY-MM-DD subfolders
    --image_base_dir PATH_TO_IMAGES \       # Required: Base directory where images are stored
    --output_dir ./manifests \               # Output directory (default: ./manifests)
    --start_date 2016-01-01 \                # Start date filter (YYYY-MM-DD)
    --end_date 2025-12-12 \                 # End date filter (YYYY-MM-DD)
    --directory_structure flat \             # 'flat' or 'nested' (default: flat)
    --lead_time 12 \                        # Lead time in hours (default: 12)
    --test_year 2025                        # Year for test set (default: 2025)
```

### Directory Structure Options

**Flat structure** (default):
- All images in one directory: `{image_base_dir}/t_z_1000_0000_12_20200101.1.png`

**Nested structure**:
- Organized by date/time: `{image_base_dir}/0000/12/2020/01/01/t_z_1000_0000_12_20200101.1.png`

### Output Files

The script creates three manifest files in `--output_dir`:

1. **`manifest_train.csv`** - Training set (2016-2024 by default)
   - Columns: `image_paths`, `target_text`
   - `image_paths`: Semicolon-separated paths to 12 images
   - `target_text`: Forecast text

2. **`manifest_test.csv`** - Test set (2025 by default)
   - Same format as training manifest

3. **`manifest_full.csv`** - Combined train + test
   - Useful for full dataset analysis

### Example Output

```csv
image_paths,target_text
./images/t_z_1000_0000_12_20200101.1.png;./images/t_z_200_0000_12_20200101.1.png;...;./images/uv_rh_850_0000_12_20200101.1.png,"Valid 12Z Wed Jan 1 2020 - 12Z Fri Jan 3 2020..."
```

### What the Script Does

1. **Scans text directory**: Finds all `.txt` files in `YYYY-MM-DD` subfolders
2. **Extracts metadata**: Determines date and init_time from file paths/names
3. **Generates image paths**: Creates paths to 12 corresponding forecast images
4. **Splits data**: Separates by year (train: 2016-2024, test: 2025)
5. **Outputs CSVs**: Creates train/test manifests

---

## Step 2: Train the Model

Use `LoRA_Training.py` to fine-tune Qwen2.5-VL with LoRA.

### Quick Start

```bash
python LoRA_Training.py \
    --train_csv ./manifests/manifest_train.csv \
    --eval_csv ./manifests/manifest_test.csv \
    --output_dir ./qwen2_5_vl_weather_lora \
    --image_size 448 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4
```

### Full Command Options

```bash
python LoRA_Training.py \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \    # Model variant (3B or 7B)
    --train_csv ./manifests/manifest_train.csv \   # Training manifest
    --eval_csv ./manifests/manifest_test.csv \     # Optional: evaluation set
    --output_dir ./qwen2_5_vl_weather_lora \        # Output directory
    --image_size 448 \                              # Optional: resize all images (e.g., 448 or 512)
    --num_train_epochs 3 \                          # Number of training epochs
    --per_device_train_batch_size 1 \               # Batch size per device
    --per_device_eval_batch_size 1 \               # Eval batch size
    --learning_rate 1e-4 \                          # Learning rate
    --warmup_ratio 0.03 \                           # Warmup steps ratio
    --weight_decay 0.01 \                           # Weight decay
    --gradient_accumulation_steps 4 \               # Effective batch = batch_size * accumulation_steps
    --fp16 \                                        # Use FP16 (if GPU supports)
    --bf16 \                                        # Use BF16 (if GPU supports)
    --logging_steps 10 \                            # Log every N steps
    --save_steps 500 \                              # Save checkpoint every N steps
    --save_total_limit 3 \                          # Keep only last N checkpoints
    --max_steps -1                                   # Limit total steps (-1 = use epochs)
```

### Model Variants

| Model | Parameters | VRAM | Speed | Quality |
|-------|-----------|------|-------|---------|
| Qwen/Qwen2.5-VL-3B-Instruct | 3B | ~6-8GB | Fast | Good |
| Qwen/Qwen2.5-VL-7B-Instruct | 7B | ~14-16GB | Medium | Better |

**Recommendation**: Start with 3B for prototyping, use 7B for final training.

### Training Features

- **LoRA fine-tuning**: Only trains a small number of parameters (efficient)
- **Multi-image support**: Handles 12 weather chart images per example
- **Automatic checkpointing**: Saves checkpoints during training
- **Evaluation**: Optional validation set evaluation
- **Logging**: TensorBoard-compatible logs
- **Gradient accumulation**: Simulate larger batch sizes

### Output

After training completes:
- **Model checkpoints**: Saved to `--output_dir` (HuggingFace-compatible format)
- **Training logs**: `{output_dir}/logs/` (view with TensorBoard)
- **Final model**: `{output_dir}/` (ready for inference)

---

## Complete Training Workflow

### Step-by-Step Example

```bash
# 1. Create manifests from your data
python create_full_image_manifest.py \
    --text_dir ./forecast_text_Cleaned \
    --image_base_dir ./images \
    --output_dir ./manifests \
    --start_date 2016-01-01 \
    --end_date 2025-12-12 \
    --test_year 2025

# 2. Verify manifests were created
ls -lh ./manifests/
# Should see: manifest_train.csv, manifest_test.csv, manifest_full.csv

# 3. Start training
python LoRA_Training.py \
    --train_csv ./manifests/manifest_train.csv \
    --eval_csv ./manifests/manifest_test.csv \
    --output_dir ./checkpoints/weather_lora \
    --model_name Qwen/Qwen2.5-VL-3B-Instruct \
    --image_size 448 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --fp16 \
    --logging_steps 10 \
    --save_steps 500

# 4. Monitor training (in another terminal)
tensorboard --logdir ./checkpoints/weather_lora/logs
```

---

## Training Tips

### 1. Batch Size Selection
- Start small (`per_device_train_batch_size=1`)
- Increase if you have GPU memory available
- Use `gradient_accumulation_steps` to simulate larger batches
- **Effective batch size** = `per_device_train_batch_size × gradient_accumulation_steps × num_gpus`

### 2. Learning Rate
- Default `1e-4` works well for LoRA
- Lower if training is unstable (loss oscillates)
- Higher if convergence is slow (loss decreases very slowly)

### 3. GPU Memory Management
-- Use smaller model (3B instead of 7B)
-- Reduce batch size
-- Use FP16/BF16 mixed precision (`--fp16` or `--bf16`)
-- Increase `gradient_accumulation_steps` to maintain effective batch size
-- **Downscale images with `--image_size` (e.g., 448)** to reduce vision-tower activations

### 4. Monitoring Training
- Watch loss values decrease over epochs
- Check TensorBoard: `tensorboard --logdir {output_dir}/logs`
- Monitor evaluation loss if using `--eval_csv`

### 5. Data Quality
- Verify image paths exist before training
- Check that `target_text` contains high-quality forecast text
- Ensure images and text are properly aligned (same date/time)

### 6. Checkpointing
- Checkpoints saved every `--save_steps` steps
- Only last `--save_total_limit` checkpoints kept
- Final model saved at end of training

---

## Troubleshooting

### Out of Memory (OOM)
```bash
# Solutions:
# 1. Downscale input images (strongest lever)
--image_size 448

# 2. Use smaller model
--model_name Qwen/Qwen2.5-VL-3B-Instruct

# 3. Reduce batch size
--per_device_train_batch_size 1

# 4. Use gradient accumulation
--gradient_accumulation_steps 8

# 5. Enable FP16
--fp16
```

### Training Loss Not Decreasing
- Check data quality (verify images and text alignment)
- Try lower learning rate (`--learning_rate 5e-5`)
- Verify labels are properly masked (prompt tokens should be -100)
- Check that images exist at specified paths

### Slow Training
- Ensure GPU is being used (check `nvidia-smi`)
- Use FP16/BF16 if supported (`--fp16` or `--bf16`)
- Increase batch size if memory allows
- Reduce `logging_steps` and `save_steps` frequency

### Import Errors
```bash
pip install --upgrade transformers peft accelerate pandas
```

### Image Not Found Errors
- Verify image paths in manifest CSV are correct
- Check `--image_base_dir` matches your actual image directory
- Ensure images follow naming pattern: `{variable}_{init_time}_{lead_time}_{date}.1.png`

### Manifest Format Issues
The script supports two manifest formats:
1. **Semicolon-separated** (from `create_full_image_manifest.py`): `image_paths`, `target_text`
2. **Individual columns**: 12 image path columns + `text` column

If you get format errors, check your CSV column names match one of these formats.

---

## Next Steps After Training

### 1. Load Fine-Tuned Model

Modify your inference script to load the LoRA adapter:

```python
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor

# Load base model
base_model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./checkpoints/weather_lora")
model.eval()
```

### 2. Evaluate on Test Set

Run inference on `manifest_test.csv` and compare against ground truth.

### 3. Compare Baselines

Benchmark fine-tuned model against:
- Pretrained Qwen2.5-VL (no fine-tuning)
- Climatology/persistence baselines

### 4. Iterate

- Adjust hyperparameters if needed
- Train for more epochs if loss still decreasing
- Try different LoRA configurations (rank, alpha)

---

## LoRA Configuration

The current LoRA configuration in `LoRA_Training.py`:

```python
LoraConfig(
    r=64,                    # LoRA rank (higher = more parameters)
    lora_alpha=128,          # LoRA alpha (scaling factor)
    lora_dropout=0.05,       # Dropout rate
    target_modules=[          # Modules to apply LoRA to
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj"
    ],
)
```

To modify, edit `LoRA_Training.py` around line 277.

---

## Additional Resources

- **HuggingFace Trainer**: https://huggingface.co/docs/transformers/training
- **PEFT/LoRA**: https://huggingface.co/docs/peft
- **Qwen2.5-VL**: https://huggingface.co/collections/Qwen/qwen25-vl
- **TensorBoard**: https://www.tensorflow.org/tensorboard

---

## Quick Reference

### Create Manifests
```bash
python create_full_image_manifest.py \
    --text_dir ./forecast_text_Cleaned \
    --image_base_dir ./images \
    --output_dir ./manifests
```

### Train Model
```bash
python LoRA_Training.py \
    --train_csv ./manifests/manifest_train.csv \
    --eval_csv ./manifests/manifest_test.csv \
    --output_dir ./checkpoints/weather_lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --fp16
```

### Monitor Training
```bash
tensorboard --logdir ./checkpoints/weather_lora/logs
```

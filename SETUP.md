# Qwen2.5-VL Weather Forecasting Setup Guide

## Overview
This project uses Qwen2.5-VL-3B-Instruct to generate text forecasts from weather chart images derived from NetCDF files.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 8GB VRAM (recommended for 3B model)
  - 3B model: ~6-8GB VRAM
  - 7B model: ~14-16GB VRAM
- **CPU**: Will work but be significantly slower
- **RAM**: 16GB+ recommended

### Software Requirements
- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- Git

## Installation

### 1. Create Virtual Environment
```bash
cd c:\Users\Daniel\WeatherLMM
python -m venv venv
venv\Scripts\activate
```

### 2. Install PyTorch with CUDA Support
Visit https://pytorch.org/ and install the appropriate version, or use:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Qwen-VL-Utils
```bash
pip install qwen-vl-utils
```

### 5. Verify Installation
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## Usage

### Basic Inference Demo

Run the main inference script:
```bash
python qwen_inference.py
```

This will:
1. Load a sample NetCDF file from your 2019/01 directory
2. Convert it to a weather chart image
3. Initialize Qwen2.5-VL-3B-Instruct
4. Generate a text forecast
5. Save the chart as `sample_weather_chart.png`

### Using Weather Utils Separately

Test the NetCDF processing utilities:
```bash
python weather_utils.py
```

### Custom Inference Example

```python
from qwen_inference import WeatherForecastInference
from weather_utils import netcdf_to_image

# Load your weather data
image, metadata = netcdf_to_image("path/to/your/file.nc")

# Initialize model
model = WeatherForecastInference(device="cuda")

# Generate forecast
forecast = model.generate_forecast(image)
print(forecast)
```

## Configuration

### Update Data Paths
Edit the paths in `qwen_inference.py`:
```python
DATA_DIR = Path(r"YOUR_PATH_HERE\2019\01")
SAMPLE_FILE = DATA_DIR / "ifs_fc_2t_0000_12_20190101.nc"
```

### Adjust Model Settings
In `WeatherForecastInference.__init__()`:
- Change `model_name` to use 7B: `"Qwen/Qwen2.5-VL-7B-Instruct"`
- Adjust `max_new_tokens` for longer/shorter outputs

### Customize Prompts
Modify the default prompt in `generate_forecast()` to:
- Focus on specific weather features
- Match your desired output format
- Include domain-specific terminology

## Model Sizes

| Model | Parameters | VRAM | Speed | Quality |
|-------|-----------|------|-------|---------|
| 3B-Instruct | 3B | ~6-8GB | Fast | Good |
| 7B-Instruct | 7B | ~14-16GB | Medium | Better |
| 32B-Instruct | 32B | ~64GB+ | Slow | Best |

**Recommendation**: Start with 3B for prototyping, move to 7B once approach is validated.

## Troubleshooting

### Out of Memory Error
- Use smaller model (3B instead of 7B)
- Reduce image resolution in `netcdf_to_image(dpi=100)`
- Reduce `max_new_tokens` in generation

### Slow Inference
- Ensure CUDA is properly installed
- Check `torch.cuda.is_available()` returns `True`
- Consider using `torch.compile()` (PyTorch 2.0+)

### Import Errors
```bash
pip install --upgrade transformers accelerate
```

### Cartopy Installation Issues
On Windows, Cartopy can be tricky. Try:
```bash
conda install -c conda-forge cartopy
```
Or use pre-built wheels from: https://www.lfd.uci.edu/~gohlke/pythonlibs/

## Next Steps

1. **Test baseline performance**: Run inference on multiple samples to see out-of-the-box quality
2. **Collect training data**: Pair weather charts with human-written forecasts
3. **Fine-tune with LoRA**: Adapt model to your specific forecast style (Phase 2)
4. **Evaluate**: Compare against climatology/persistence baselines

## File Structure

```
WeatherLMM/
├── qwen_inference.py       # Main inference script
├── weather_utils.py        # NetCDF processing utilities
├── requirements.txt        # Python dependencies
├── SETUP.md               # This file
├── open_netcdf.py         # Original NetCDF exploration script
└── Test.py                # Scratch file
```

## Resources

- [Qwen2.5-VL Documentation](https://huggingface.co/collections/Qwen/qwen25-vl)
- [LoRA Fine-tuning Guide](https://github.com/microsoft/LoRA)
- [ECMWF Charts](https://charts.ecmwf.int/)

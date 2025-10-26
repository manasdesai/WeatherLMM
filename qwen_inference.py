"""
Qwen2.5-VL Weather Forecast Inference Script

This script loads the Qwen2.5-VL-3B-Instruct model and generates text forecasts
from weather chart images created from NetCDF (.nc) files.
"""

# Fix OpenMP conflict between Anaconda and PyTorch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from pathlib import Path
import io


class WeatherForecastInference:
    """Inference wrapper for Qwen2.5-VL on weather forecast images."""

    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", device="cuda"):
        """
        Initialize the Qwen2.5-VL model for inference.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on ('cuda' or 'cpu')
        """
        print(f"Loading {model_name}...")

        # Load model with appropriate settings for VL tasks
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.device = device

        print(f"Model loaded successfully on {device}")

    def generate_forecast(self, image, prompt=None, max_new_tokens=512):
        """
        Generate a text forecast from a weather chart image.

        Args:
            image: PIL Image or path to image file
            prompt: Custom prompt (default: weather forecast prompt)
            max_new_tokens: Maximum length of generated text

        Returns:
            Generated forecast text
        """
        if prompt is None:
            prompt = (
                "You are a meteorologist analyzing a weather forecast chart. "
                "Describe the temperature patterns, geographic distribution, "
                "and any notable weather features visible in this 2-meter temperature forecast map. "
                "Provide a clear, professional weather forecast based on this data."
            )

        # Prepare the conversation format Qwen2.5-VL expects
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if self.device == "cuda":
            inputs = inputs.to(self.device)

        # Generate forecast
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic for consistent forecasts
            )

        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text


def load_netcdf_as_image(nc_file_path, figsize=(10, 6), dpi=150):
    """
    Load a NetCDF file and convert it to a PIL Image for model input.

    Args:
        nc_file_path: Path to .nc file
        figsize: Figure size for the plot
        dpi: Resolution of the output image

    Returns:
        PIL.Image: Weather chart as image
        dict: Metadata (time, min/max temp, etc.)
    """
    # Load NetCDF file
    ds = xr.open_dataset(nc_file_path)

    # Extract temperature and convert to Celsius
    t2m = ds.t2m - 273.15

    # Get metadata
    time_str = pd.to_datetime(ds.time.values).strftime('%Y-%m-%d %H:%M UTC')
    temp_min = float(t2m.min().values)
    temp_max = float(t2m.max().values)
    temp_mean = float(t2m.mean().values)

    metadata = {
        'time': time_str,
        'temp_min': temp_min,
        'temp_max': temp_max,
        'temp_mean': temp_mean,
        'file': nc_file_path
    }

    # Create figure
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    # Plot temperature data
    mesh = t2m.plot(
        ax=ax,
        cmap='RdBu_r',
        add_colorbar=False,
        vmin=temp_min,
        vmax=temp_max
    )

    # Add map features
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.set_aspect(1.5)

    # Add title and colorbar
    ax.set_title(f'12-hr 2m Temperature Forecast (°C)\nValid: {time_str}', fontsize=12)
    cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.set_label('°C')

    plt.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf).convert('RGB')

    plt.close(fig)
    ds.close()

    return image, metadata


def main():
    """Demo script showing Qwen2.5-VL inference on weather data."""

    # Configuration
    DATA_DIR = Path(r"C:\Users\Daniel\Downloads\forecast_2t_12hr_2016_2024-20251026T174311Z-1-001\forecast_2t_12hr_2016_2024\2019\01")
    SAMPLE_FILE = DATA_DIR / "ifs_fc_2t_0000_12_20190101.nc"

    # Check if sample file exists
    if not SAMPLE_FILE.exists():
        print(f"Error: Sample file not found at {SAMPLE_FILE}")
        print("Please update DATA_DIR and SAMPLE_FILE paths in the script.")
        return

    print("="*80)
    print("Qwen2.5-VL Weather Forecast Inference Demo")
    print("="*80)
    print()

    # Step 1: Load NetCDF and convert to image
    print("Step 1: Loading NetCDF file and generating weather chart...")
    weather_image, metadata = load_netcdf_as_image(SAMPLE_FILE)
    print(f"  ✓ Loaded: {metadata['file']}")
    print(f"  ✓ Valid time: {metadata['time']}")
    print(f"  ✓ Temperature range: {metadata['temp_min']:.1f}°C to {metadata['temp_max']:.1f}°C")
    print(f"  ✓ Mean temperature: {metadata['temp_mean']:.1f}°C")
    print()

    # Optionally save the image for inspection
    output_image_path = "sample_weather_chart.png"
    weather_image.save(output_image_path)
    print(f"  ✓ Saved weather chart to: {output_image_path}")
    print()

    # Step 2: Initialize Qwen2.5-VL model
    print("Step 2: Initializing Qwen2.5-VL-3B-Instruct...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    if device == "cpu":
        print("  ⚠ Warning: Running on CPU. Inference will be slower.")

    inference = WeatherForecastInference(device=device)
    print()

    # Step 3: Generate forecast
    print("Step 3: Generating forecast from weather chart...")
    print("-" * 80)

    forecast_text = inference.generate_forecast(weather_image)

    print("GENERATED FORECAST:")
    print(forecast_text)
    print("-" * 80)
    print()

    # Step 4: Try with a custom prompt
    print("Step 4: Testing with custom prompt...")
    custom_prompt = (
        "Analyze this temperature forecast map and provide: "
        "1) The warmest and coldest regions, "
        "2) Overall temperature gradient patterns, "
        "3) A brief forecast summary."
    )

    print(f"Custom prompt: {custom_prompt}")
    print("-" * 80)

    custom_forecast = inference.generate_forecast(weather_image, prompt=custom_prompt)

    print("GENERATED FORECAST (Custom Prompt):")
    print(custom_forecast)
    print("-" * 80)
    print()

    print("="*80)
    print("Demo completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()

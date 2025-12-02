"""
Utility functions for processing weather NetCDF files and preparing them
for vision-language model inference.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, List
import io
import argparse


def find_netcdf_files(base_dir: str, year: int = None, month: int = None) -> List[Path]:
    """
    Find all NetCDF files in the data directory.

    Args:
        base_dir: Base directory containing year/month subdirectories
        year: Optional year filter (e.g., 2019)
        month: Optional month filter (1-12)

    Returns:
        List of Path objects to .nc files
    """
    base_path = Path(base_dir)

    if year and month:
        # Specific year and month
        pattern = f"{year}/{month:02d}/*.nc"
    elif year:
        # All months for a specific year
        pattern = f"{year}/*/*.nc"
    else:
        # All files
        pattern = "*/*/*.nc"

    nc_files = sorted(base_path.glob(pattern))
    return nc_files


def load_netcdf_data(nc_file_path: str) -> Tuple[xr.DataArray, Dict]:
    """
    Load temperature data from NetCDF file and extract metadata.

    Args:
        nc_file_path: Path to .nc file

    Returns:
        temperature_data: xarray DataArray with temperature in Celsius
        metadata: Dictionary with time, min/max temps, etc.
    """
    ds = xr.open_dataset(nc_file_path)

    # Convert temperature from Kelvin to Celsius
    t2m = ds.t2m - 273.15

    # Extract metadata
    time_str = pd.to_datetime(ds.time.values).strftime('%Y-%m-%d %H:%M UTC')

    metadata = {
        'time': time_str,
        'time_raw': ds.time.values,
        'temp_min': float(t2m.min().values),
        'temp_max': float(t2m.max().values),
        'temp_mean': float(t2m.mean().values),
        'temp_std': float(t2m.std().values),
        'lat_range': (float(t2m.latitude.min()), float(t2m.latitude.max())),
        'lon_range': (float(t2m.longitude.min()), float(t2m.longitude.max())),
        'file_path': str(nc_file_path)
    }

    return t2m, metadata


def create_weather_chart(
    temperature_data: xr.DataArray,
    title: str = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 150,
    cmap: str = 'RdBu_r',
    add_features: bool = True
) -> plt.Figure:
    """
    Create a matplotlib figure from temperature data.

    Args:
        temperature_data: xarray DataArray with temperature
        title: Plot title
        figsize: Figure size
        dpi: Resolution
        cmap: Colormap name
        add_features: Whether to add coastlines/borders

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={'projection': ccrs.PlateCarree()},
        dpi=dpi
    )

    # Plot temperature
    mesh = temperature_data.plot(
        ax=ax,
        cmap=cmap,
        add_colorbar=False,
    )

    # Add geographic features
    if add_features:
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.7)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

    ax.set_aspect(1.5)

    # Set title
    if title:
        ax.set_title(title, fontsize=12, pad=10)

    # Add colorbar
    cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.set_label('Temperature (°C)', fontsize=10)

    plt.tight_layout()

    return fig


def fig_to_pil_image(fig: plt.Figure) -> Image.Image:
    """
    Convert matplotlib figure to PIL Image.

    Args:
        fig: matplotlib Figure

    Returns:
        PIL Image in RGB format
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    plt.close(fig)

    return image


def netcdf_to_image(
    nc_file_path: str,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 150
) -> Tuple[Image.Image, Dict]:
    """
    Complete pipeline: Load NetCDF file and convert to PIL Image.

    Args:
        nc_file_path: Path to .nc file
        figsize: Figure size
        dpi: Resolution

    Returns:
        image: PIL Image of weather chart
        metadata: Dictionary with forecast metadata
    """
    # Load data
    t2m, metadata = load_netcdf_data(nc_file_path)

    # Create title
    title = f"12-hr 2m Temperature Forecast\nValid: {metadata['time']}"

    # Create chart
    fig = create_weather_chart(t2m, title=title, figsize=figsize, dpi=dpi)

    # Convert to image
    image = fig_to_pil_image(fig)

    return image, metadata


def batch_process_netcdf_files(
    nc_files: List[Path],
    output_dir: str = None,
    save_images: bool = False
) -> List[Tuple[Image.Image, Dict]]:
    """
    Process multiple NetCDF files into images.

    Args:
        nc_files: List of paths to .nc files
        output_dir: Directory to save images (if save_images=True)
        save_images: Whether to save images to disk

    Returns:
        List of (image, metadata) tuples
    """
    results = []

    if save_images and output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    for nc_file in nc_files:
        try:
            image, metadata = netcdf_to_image(str(nc_file))
            results.append((image, metadata))

            if save_images and output_dir:
                # Save with timestamp-based filename
                time_str = pd.to_datetime(metadata['time_raw']).strftime('%Y%m%d_%H%M')
                save_path = output_path / f"forecast_{time_str}.png"
                image.save(save_path)
                print(f"Saved: {save_path}")

        except Exception as e:
            print(f"Error processing {nc_file}: {e}")
            continue

    return results


if __name__ == "__main__":
    #Example: Process a single file (commented out)
    
    sample_file = r"C:\Users\Daniel\Downloads\forecast_2t_12hr_2016_2024-20251026T174311Z-1-001\forecast_2t_12hr_2016_2024\2019\01\ifs_fc_2t_0000_12_20190101.nc"
    
    if Path(sample_file).exists():
        print(f"\nProcessing: {sample_file}")
        image, metadata = netcdf_to_image(sample_file)

        print("\nMetadata:")
        for key, value in metadata.items():
            if key != 'time_raw':
                print(f"  {key}: {value}")

        # Save for inspection
        image.save("demo_weather_chart.png")
        print("\n✓ Saved demo_weather_chart.png")
    else:
        print(f"\nSample file not found: {sample_file}")
        print("Update the path in the script to test.")
    

    # parser = argparse.ArgumentParser(description='Convert NetCDF weather files to PNG images')
    # parser.add_argument('--directory', '-d', type=str, required=True,
    #                    help='Root directory to search for .nc files')
    # parser.add_argument('--figsize', type=int, nargs=2, default=[10, 6],
    #                    help='Figure size (width height), default: 10 6')
    # parser.add_argument('--dpi', type=int, default=150,
    #                    help='Image resolution (DPI), default: 150')
    # args = parser.parse_args()

    # print("Weather Utils - NetCDF to PNG Converter")
    # print("=" * 60)
    # print(f"Searching directory: {args.directory}")
    # print(f"Figure size: {args.figsize[0]}x{args.figsize[1]}")
    # print(f"DPI: {args.dpi}")
    # print()

    # # Find all .nc files recursively
    # root_path = Path(args.directory)
    # if not root_path.exists():
    #     print(f"Error: Directory not found: {args.directory}")
    #     exit(1)

    # nc_files = list(root_path.rglob("*.nc"))

    # if not nc_files:
    #     print(f"No .nc files found in {args.directory}")
    #     exit(0)

    # print(f"Found {len(nc_files)} NetCDF files")
    # print("=" * 60)
    # print()

    # # Process each file
    # successful = 0
    # failed = 0

    # for i, nc_file in enumerate(nc_files, 1):
    #     try:
    #         # Generate image
    #         image, metadata = netcdf_to_image(
    #             str(nc_file),
    #             figsize=tuple(args.figsize),
    #             dpi=args.dpi
    #         )

    #         # Save in the same directory as the .nc file
    #         output_path = nc_file.with_suffix('.png')
    #         image.save(output_path)

    #         print(f"[{i}/{len(nc_files)}] ✓ {nc_file.name} -> {output_path.name}")
    #         successful += 1

    #     except Exception as e:
    #         print(f"[{i}/{len(nc_files)}] ✗ {nc_file.name} - Error: {e}")
    #         failed += 1

    # print()
    # print("=" * 60)
    # print(f"Conversion complete!")
    # print(f"  Successful: {successful}")
    # print(f"  Failed: {failed}")
    # print(f"  Total: {len(nc_files)}")
    # print("=" * 60)
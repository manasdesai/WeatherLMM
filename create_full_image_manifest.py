"""
Create a full manifest from forecast text directory (2016-2025).

This script:
1. Scans the text directory (forecast_text_Cleaned) for available text files
2. Determines date and init_time from text file location and name
3. Generates 12 image paths based on the naming pattern (doesn't verify existence)
4. Generates train (2016-2024) and test (2025) splits
5. Outputs manifests with image_paths (semicolon-separated) and target_text

Text directory structure: {text_dir}/YYYY-MM-DD/HH-MM.txt
Image naming pattern: {variable}_{init_time}_{lead_time}_{date}.1.png
Variables: t_z_1000, t_z_200, t_z_500, t_z_700, t_z_850, t2m_wind10m, 
           thickness_mslp, uv_rh_1000, uv_rh_200, uv_rh_500, uv_rh_700, uv_rh_850

Usage:
  python create_full_image_manifest.py \
    --text_dir /path/to/forecast_text_Cleaned \
    --image_base_dir /path/to/images \
    --output_dir ./manifests \
    --start_date 2016-01-01 \
    --end_date 2025-12-12
"""

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


# The 12 forecast image types
IMAGE_VARIABLES = [
    't_z_1000',
    't_z_200',
    't_z_500',
    't_z_700',
    't_z_850',
    't2m_wind10m',
    'thickness_mslp',
    'uv_rh_1000',
    'uv_rh_200',
    'uv_rh_500',
    'uv_rh_700',
    'uv_rh_850',
]


def hour_from_txt_name(txt_name: str) -> Optional[int]:
    """
    Extract hour from text filename.
    
    Example: '03-55.txt' -> 3, '19-31.txt' -> 19
    """
    base = Path(txt_name).stem
    if '-' not in base:
        return None
    prefix = base.split('-')[0]
    try:
        hour = int(prefix)
        return hour
    except ValueError:
        return None


def generate_image_paths(
    date: str,
    init_time: str,
    lead_time: str,
    image_base_dir: str,
    directory_structure: str = "flat"
) -> List[str]:
    """
    Generate paths to all 12 forecast images for a given date/time combination.
    
    Args:
        date: YYYYMMDD format (e.g., '20200101')
        init_time: HHMM format (e.g., '0000')
        lead_time: Lead time in hours (e.g., '12')
        image_base_dir: Base directory where images are stored
        directory_structure: 'flat' (all images in one dir) or 'nested' (organized by date/time)
    
    Returns:
        List of 12 image paths
    """
    year = date[0:4]
    month = date[4:6]
    day = date[6:8]
    
    image_paths = []
    
    if directory_structure == "nested":
        # Structure: {base_dir}/{init_time}/{lead_time}/{year}/{month}/{day}/
        # Example: ifs_fc_images_20160101_20251212/0000/12/2020/01/01/
        image_dir = Path(image_base_dir) / init_time / lead_time / year / month / day
    else:
        # Flat structure: all images in base_dir
        image_dir = Path(image_base_dir)
    
    for variable in IMAGE_VARIABLES:
        # Image naming: {variable}_{init_time}_{lead_time}_{date}.1.png
        image_name = f"{variable}_{init_time}_{lead_time}_{date}.1.png"
        image_path = image_dir / image_name
        image_paths.append(str(image_path))
    
    return image_paths


def scan_text_directory(
    text_dir: Path,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    lead_time: str = "12"
) -> List[Tuple[str, str, str]]:
    """
    Scan text directory and extract (date, init_time, text_content) tuples.
    
    Args:
        text_dir: Base directory containing YYYY-MM-DD subfolders
        start_date: Optional start date filter
        end_date: Optional end date filter
        lead_time: Lead time in hours (default: '12')
    
    Returns:
        List of (date_YYYYMMDD, init_time, text_content) tuples
    """
    records = []
    
    for date_dir in sorted(text_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        
        # Parse date from directory name (YYYY-MM-DD)
        try:
            date_dt = datetime.strptime(date_dir.name, '%Y-%m-%d')
        except ValueError:
            continue
        
        # Apply date filters
        if start_date and date_dt < start_date:
            continue
        if end_date and date_dt > end_date:
            continue
        
        # Convert to YYYYMMDD format
        date_str = date_dt.strftime('%Y%m%d')
        
        # Process text files in this date directory
        for txt_file in sorted(date_dir.glob('*.txt')):
            hour = hour_from_txt_name(txt_file.name)
            if hour is None:
                continue
            
            # Determine init_time: morning (hour < 12) -> 0000, evening (hour >= 12) -> 1200
            init_time = '0000' if hour < 12 else '1200'
            
            # Read text content
            try:
                content = txt_file.read_text(encoding='utf-8').strip()
            except Exception:
                try:
                    content = txt_file.read_text(encoding='latin-1').strip()
                except Exception:
                    continue
            
            if content:  # Only add if we have content
                records.append((date_str, init_time, content))
    
    return records


def create_manifest(
    text_dir: str,
    image_base_dir: str,
    output_dir: str,
    start_date: str = "2016-01-01",
    end_date: str = "2025-12-12",
    directory_structure: str = "flat",
    lead_time: str = "12",
    test_year: int = 2025
):
    """
    Create full manifest with train/test split.
    
    Args:
        text_dir: Directory containing text files (YYYY-MM-DD subfolders)
        image_base_dir: Base directory where images are stored (paths will be generated)
        output_dir: Directory to write output manifests
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        directory_structure: 'flat' or 'nested' (for image paths)
        lead_time: Lead time in hours (default: '12')
        test_year: Year to use for test set (default: 2025)
    """
    text_path = Path(text_dir)
    if not text_path.exists():
        raise ValueError(f"Text directory does not exist: {text_dir}")
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    print(f"Scanning text directory: {text_dir}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Image directory structure: {directory_structure}")
    print(f"Lead time: {lead_time} hours")
    
    # Scan text directory
    text_records = scan_text_directory(text_path, start_dt, end_dt, lead_time)
    
    print(f"Found {len(text_records)} text files")
    
    # Process each text record and generate image paths
    train_rows = []
    test_rows = []
    
    for date, init_time, target_text in text_records:
        # Generate 12 image paths based on pattern
        image_paths = generate_image_paths(
            date, init_time, lead_time, image_base_dir, directory_structure
        )
        image_paths_str = ';'.join(image_paths)
        
        row = {
            'image_paths': image_paths_str,
            'target_text': target_text
        }
        
        # Split by year
        year = int(date[0:4])
        if year == test_year:
            test_rows.append(row)
        else:
            train_rows.append(row)
    
    # Write output manifests
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_csv = output_path / 'manifest_train.csv'
    test_csv = output_path / 'manifest_test.csv'
    full_csv = output_path / 'manifest_full.csv'
    
    # Write train manifest
    with open(train_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['image_paths', 'target_text'])
        writer.writeheader()
        writer.writerows(train_rows)
    
    # Write test manifest
    with open(test_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['image_paths', 'target_text'])
        writer.writeheader()
        writer.writerows(test_rows)
    
    # Write full manifest
    with open(full_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['image_paths', 'target_text'])
        writer.writeheader()
        writer.writerows(train_rows + test_rows)
    
    print(f"\nManifest creation complete:")
    print(f"  Train records: {len(train_rows)}")
    print(f"  Test records: {len(test_rows)}")
    print(f"  Total records: {len(train_rows) + len(test_rows)}")
    print(f"\nOutput files:")
    print(f"  Train: {train_csv}")
    print(f"  Test: {test_csv}")
    print(f"  Full: {full_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='Create full manifest from forecast text directory (2016-2025)'
    )
    parser.add_argument(
        '--text_dir',
        required=True,
        help='Directory containing text files (forecast_text_Cleaned with YYYY-MM-DD subfolders)'
    )
    parser.add_argument(
        '--image_base_dir',
        required=True,
        help='Base directory where images are stored (paths will be generated)'
    )
    parser.add_argument(
        '--output_dir',
        default='./manifests',
        help='Directory to write output manifests'
    )
    parser.add_argument(
        '--start_date',
        default='2016-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end_date',
        default='2025-12-12',
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--directory_structure',
        choices=['flat', 'nested'],
        default='flat',
        help='Image directory structure: flat (all in one dir) or nested (organized by date/time)'
    )
    parser.add_argument(
        '--lead_time',
        default='12',
        help='Lead time in hours (default: 12)'
    )
    parser.add_argument(
        '--test_year',
        type=int,
        default=2025,
        help='Year to use for test set (default: 2025)'
    )
    args = parser.parse_args()
    
    create_manifest(
        args.text_dir,
        args.image_base_dir,
        args.output_dir,
        args.start_date,
        args.end_date,
        args.directory_structure,
        args.lead_time,
        args.test_year
    )


if __name__ == '__main__':
    main()


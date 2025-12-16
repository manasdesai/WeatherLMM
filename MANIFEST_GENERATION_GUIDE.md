# Manifest Generation Guide

This guide explains how to generate the training manifest files for the WeatherLMM project using the forecast images and text data.

## Overview

The `create_full_image_manifest.py` script creates CSV manifest files that map 12 forecast images to their corresponding text descriptions. The manifest is automatically split into train (2016-2024) and test (2025) sets.

## Prerequisites

### Required Data

1. **Text Data**: `forecast_text_Cleaned/forecast_text/` directory
   - Structure: `forecast_text_Cleaned/forecast_text/YYYY-MM-DD/*.txt`
   - Each date folder contains text files named like `03-55.txt` or `19-31.txt`
   - The hour prefix (03, 19) indicates morning (< 12) or evening (>= 12) forecasts

2. **Image Data**: `ifs_fc_images_20160101_20251212/` directory
   - Structure: `ifs_fc_images_20160101_20251212/{init_time}/12/{year}/{month}/{day}/*.png`
   - Where `{init_time}` is `0000` or `1200`
   - Each date/time combination should have 12 images:
     - `t_z_1000_{init_time}_12_{date}.1.png`
     - `t_z_200_{init_time}_12_{date}.1.png`
     - `t_z_500_{init_time}_12_{date}.1.png`
     - `t_z_700_{init_time}_12_{date}.1.png`
     - `t_z_850_{init_time}_12_{date}.1.png`
     - `t2m_wind10m_{init_time}_12_{date}.1.png`
     - `thickness_mslp_{init_time}_12_{date}.1.png`
     - `uv_rh_1000_{init_time}_12_{date}.1.png`
     - `uv_rh_200_{init_time}_12_{date}.1.png`
     - `uv_rh_500_{init_time}_12_{date}.1.png`
     - `uv_rh_700_{init_time}_12_{date}.1.png`
     - `uv_rh_850_{init_time}_12_{date}.1.png`

### Date Range

The script processes data from **2016-01-01** to **2025-12-12**.

## Step-by-Step Instructions

### 1. Verify Data Structure

Before running the script, verify your data is organized correctly:

```bash
# Check text directory structure
ls forecast_text_Cleaned/forecast_text/ | head -5
# Should show: 2016-01-01, 2016-01-02, etc.

# Check a sample text directory
ls forecast_text_Cleaned/forecast_text/2016-01-01/
# Should show text files like: 03-55.txt, 19-31.txt

# Check image directory structure
ls ifs_fc_images_20160101_20251212/
# Should show: 0000/, 1200/

# Check nested structure
ls ifs_fc_images_20160101_20251212/0000/12/2016/01/01/
# Should show 12 PNG files
```

### 2. Check Data Availability

Verify you have data for the expected date range:

```bash
# Count text files (should be thousands)
find forecast_text_Cleaned/forecast_text -name "*.txt" | wc -l

# Count image directories (each should have 12 images)
find ifs_fc_images_20160101_20251212 -type d -path "*/12/*/*/*" | wc -l

# Sample check: verify a specific date has images
ls ifs_fc_images_20160101_20251212/0000/12/2020/01/01/*.png | wc -l
# Should return 12
```

### 3. Run the Script

```bash
python create_full_image_manifest.py \
  --text_dir ./forecast_text_Cleaned/forecast_text \
  --image_base_dir ./ifs_fc_images_20160101_20251212 \
  --output_dir ./manifests \
  --start_date 2016-01-01 \
  --end_date 2025-12-12 \
  --directory_structure nested \
  --lead_time 12
```

### 4. Verify Output

After running, check the output:

```bash
# Check that manifests directory was created
ls manifests/
# Should show: manifest_train.csv, manifest_test.csv, manifest_full.csv

# Check record counts
wc -l manifests/*.csv
# manifest_train.csv should have ~6,500+ records
# manifest_test.csv should have ~600+ records
# manifest_full.csv should be the sum

# Verify a sample row
head -2 manifests/manifest_train.csv
# Should show header and one data row with 12 image paths (semicolon-separated)
```

### 5. Sample Verification

Verify the paths are correct:

```bash
# Check first few image paths in train set
python -c "
import csv
with open('manifests/manifest_train.csv') as f:
    reader = csv.DictReader(f)
    row = next(reader)
    paths = row['image_paths'].split(';')
    print('Number of images:', len(paths))
    print('First path:', paths[0])
    print('Last path:', paths[-1])
    print('Text length:', len(row['target_text']))
"
```

Expected output:
- Number of images: 12
- First path should match: `ifs_fc_images_20160101_20251212/0000/12/2016/01/01/t_z_1000_0000_12_20160101.1.png`
- Text should be non-empty

## Understanding the Output

### Manifest Format

Each CSV file has two columns:

1. **`image_paths`**: Semicolon-separated list of 12 image file paths
   - Example: `path1;path2;path3;...;path12`
   - Always contains exactly 12 paths in a fixed order

2. **`target_text`**: The corresponding forecast text description
   - Multi-line text describing the weather forecast
   - May be empty if no matching text file was found

### Train/Test Split

- **Train set** (`manifest_train.csv`): All records from 2016-2024
- **Test set** (`manifest_test.csv`): All records from 2025
- **Full set** (`manifest_full.csv`): Combined train + test

### Expected Record Counts

Based on the date range (2016-01-01 to 2025-12-12):
- Approximately **6,500-7,000** train records
- Approximately **600-700** test records
- Total: **7,000-8,000** records

*Note: Actual counts depend on data availability*

## Troubleshooting

### Issue: "Text directory does not exist"

**Solution**: Check the path to your text directory. Use absolute paths if needed:
```bash
python create_full_image_manifest.py \
  --text_dir /full/path/to/forecast_text_Cleaned/forecast_text \
  ...
```

### Issue: "Found 0 text files"

**Possible causes**:
1. Date range doesn't match available data
2. Directory structure is incorrect
3. Text files are in a different location

**Solution**: 
```bash
# Check what dates you actually have
ls forecast_text_Cleaned/forecast_text/ | head -10

# Adjust start_date and end_date to match your data
```

### Issue: Image paths don't exist

**Note**: The script generates paths based on the pattern - it doesn't verify images exist. This is intentional since you may not have all images downloaded yet.

**To verify paths are correct**:
```bash
# Check if a sample path structure matches
ls -d ifs_fc_images_20160101_20251212/0000/12/2016/01/01/
# Should exist if you have that date's images
```

### Issue: Missing images for some dates

**This is normal**: The script will generate paths for all text files found, even if corresponding images don't exist yet. The training code should handle missing images gracefully.

### Issue: Empty target_text

**Possible causes**:
1. Text file couldn't be read (encoding issue)
2. No matching text file found for that date/time

**Solution**: Check if text files exist for those dates:
```bash
ls forecast_text_Cleaned/forecast_text/2016-01-01/
```

## Script Parameters

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| `--text_dir` | Path to text directory | - | Yes |
| `--image_base_dir` | Base directory for images | - | Yes |
| `--output_dir` | Where to write manifests | `./manifests` | No |
| `--start_date` | Start date (YYYY-MM-DD) | `2016-01-01` | No |
| `--end_date` | End date (YYYY-MM-DD) | `2025-12-12` | No |
| `--directory_structure` | `flat` or `nested` | `flat` | No |
| `--lead_time` | Forecast lead time (hours) | `12` | No |
| `--test_year` | Year for test set | `2025` | No |

## How It Works

1. **Scans text directory**: Finds all `YYYY-MM-DD` folders and `.txt` files
2. **Extracts metadata**: 
   - Date from folder name
   - Init time (`0000` or `1200`) from text filename hour
3. **Generates image paths**: Creates 12 image paths using the pattern:
   ```
   {image_base_dir}/{init_time}/12/{year}/{month}/{day}/{variable}_{init_time}_12_{date}.1.png
   ```
4. **Reads text content**: Loads the forecast text from the `.txt` file
5. **Splits by year**: 2025 → test, 2016-2024 → train
6. **Writes CSVs**: Creates three manifest files

## Next Steps

After generating the manifests:

1. **Verify paths**: Spot-check a few image paths to ensure they're correct
2. **Check data quality**: Review a few text samples to ensure they're readable
3. **Use for training**: The manifests are ready to use with `LoRA_Training.py` or `train.py`

## Questions?

If you encounter issues:
1. Check the error message carefully
2. Verify your directory structure matches the expected format
3. Check that you have data for the specified date range
4. Review the troubleshooting section above


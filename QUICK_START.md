# Quick Start: Generate Manifest

## TL;DR

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

## What You Need

1. **Text data**: `forecast_text_Cleaned/forecast_text/` with `YYYY-MM-DD/` subfolders
2. **Image data**: `ifs_fc_images_20160101_20251212/` with structure `{init_time}/12/{year}/{month}/{day}/*.png`

## What You Get

Three CSV files in `./manifests/`:
- `manifest_train.csv` - Training data (2016-2024)
- `manifest_test.csv` - Test data (2025)
- `manifest_full.csv` - Combined dataset

Each row has:
- `image_paths`: 12 semicolon-separated image paths
- `target_text`: Forecast text description

## Verify It Worked

```bash
# Check files were created
ls manifests/*.csv

# Check record counts
wc -l manifests/*.csv

# Sample a row
head -2 manifests/manifest_train.csv
```

## Full Documentation

See `MANIFEST_GENERATION_GUIDE.md` for detailed instructions, troubleshooting, and explanations.


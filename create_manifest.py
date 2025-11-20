"""
Create a CSV manifest from dataset organized by date folders.

Folder layout expected:
  /path/to/data/2020-01-31/
    03-55.txt
    19-31.txt
    ifs_fc_sfc_0000_12_20200131.nc
    ifs_fc_sfc_1200_12_20200131.nc

Rules:
 - If the hour prefix of the txt filename is in [0,12) -> match morning file containing '0000'
 - If hour in [12,24) -> match evening file containing '1200'

Output CSV columns: nc_path,prompt,target_text

Usage:
  python create_manifest.py --data_dir /path/to/data --output manifest.csv
"""

import argparse
import csv
from pathlib import Path
from typing import Optional


DEFAULT_PROMPT = (
    "You are a meteorologist analyzing a 2-meter temperature forecast map. "
    "Describe temperature patterns, regions of interest, and provide a concise forecast."
)


def find_matching_nc(date_dir: Path, is_morning: bool) -> Optional[Path]:
    """Search for a .nc file containing '0000' or '1200' in the filename in the given directory."""
    target = '0000' if is_morning else '1200'
    for nc in date_dir.glob('*.nc'):
        name = nc.name
        if target in name:
            return nc
    return None


def hour_from_txt_name(txt_name: str) -> Optional[int]:
    # txt_name like '03-55.txt' or '3-55.txt'
    base = Path(txt_name).stem
    if '-' not in base:
        return None
    prefix = base.split('-')[0]
    try:
        hour = int(prefix)
        return hour
    except ValueError:
        return None


def create_manifest(data_dir: str, output_csv: str, prompt_template: str = DEFAULT_PROMPT):
    base = Path(data_dir)
    rows = []

    for date_dir in sorted(base.iterdir()):
        if not date_dir.is_dir():
            continue
        # Expect directory name like YYYY-MM-DD
        for txt in sorted(date_dir.glob('*.txt')):
            hour = hour_from_txt_name(txt.name)
            if hour is None:
                print(f"Skipping text file with unexpected name: {txt}")
                continue

            is_morning = hour < 12
            nc = find_matching_nc(date_dir, is_morning)
            if nc is None:
                print(f"Warning: no matching .nc for {txt} (morning={is_morning}) in {date_dir}")
                continue

            # Read text content
            try:
                content = txt.read_text(encoding='utf-8').strip()
            except Exception:
                content = txt.read_text(encoding='latin-1').strip()

            # We do not include per-example prompts in the manifest; training
            # will use a fixed internal prompt. Only save the nc_path and target_text.
            print(f"Adding row: nc={nc}, target_text_len={len(content)}")
            rows.append({'nc_path': str(nc.resolve()), 'target_text': content})

    # Write CSV
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['nc_path', 'target_text'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Root dataset dir with YYYY-MM-DD subfolders')
    parser.add_argument('--output', required=True, help='Output CSV manifest path')
    parser.add_argument('--prompt', default=DEFAULT_PROMPT, help='Prompt template to include in manifest')
    args = parser.parse_args()

    create_manifest(args.data_dir, args.output, args.prompt)


if __name__ == '__main__':
    main()

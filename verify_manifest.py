#!/usr/bin/env python3
"""
Quick verification script for generated manifests.

Usage:
    python verify_manifest.py manifests/manifest_train.csv
    python verify_manifest.py manifests/manifest_test.csv
"""

import argparse
import csv
import sys
from pathlib import Path


def verify_manifest(manifest_path: str):
    """Verify a manifest file and print statistics."""
    path = Path(manifest_path)
    
    if not path.exists():
        print(f"ERROR: Manifest file not found: {manifest_path}")
        sys.exit(1)
    
    print(f"Verifying: {manifest_path}\n")
    
    total_records = 0
    records_with_text = 0
    records_without_text = 0
    image_path_counts = []
    sample_paths = []
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader):
            total_records += 1
            
            # Check image paths
            image_paths = row.get('image_paths', '').split(';')
            image_path_counts.append(len(image_paths))
            
            if i < 3:  # Save first few paths
                sample_paths.append(image_paths)
            
            # Check text
            target_text = row.get('target_text', '').strip()
            if target_text:
                records_with_text += 1
            else:
                records_without_text += 1
    
    # Print statistics
    print(f"Total records: {total_records}")
    print(f"Records with text: {records_with_text}")
    print(f"Records without text: {records_without_text}")
    print(f"\nImage paths per record:")
    print(f"  Min: {min(image_path_counts)}")
    print(f"  Max: {max(image_path_counts)}")
    print(f"  Expected: 12")
    
    if min(image_path_counts) != 12 or max(image_path_counts) != 12:
        print("  ⚠️  WARNING: Not all records have exactly 12 image paths!")
    else:
        print("  ✓ All records have exactly 12 image paths")
    
    # Show sample paths
    if sample_paths:
        print(f"\nSample image paths (first record):")
        for i, img_path in enumerate(sample_paths[0][:3], 1):
            print(f"  {i}. {img_path}")
        if len(sample_paths[0]) > 3:
            print(f"  ... ({len(sample_paths[0]) - 3} more)")
    
    # Check path structure
    if sample_paths and sample_paths[0]:
        first_path = sample_paths[0][0]
        if 'ifs_fc_images' in first_path:
            print(f"\n✓ Path structure looks correct")
            print(f"  Base: {first_path.split('/')[0]}")
        else:
            print(f"\n⚠️  Path structure may be incorrect")
            print(f"  First path: {first_path}")
    
    print()


def main():
    parser = argparse.ArgumentParser(description='Verify manifest files')
    parser.add_argument(
        'manifest',
        help='Path to manifest CSV file to verify'
    )
    args = parser.parse_args()
    
    verify_manifest(args.manifest)


if __name__ == '__main__':
    main()


"""
Visualize evaluation results with images, text comparisons, and statistics.

This script creates an interactive HTML report showing:
- Original images and text vs predicted text
- Full statistical breakdown of metrics
- Season-by-season analysis

Usage:
    python visualize_evaluation.py \
        --results_csv ./evaluation_results/detailed_results.csv \
        --test_csv ./manifests/manifest_test.csv \
        --output_dir ./evaluation_visualization
"""

import os
import re
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from PIL import Image
import base64
from io import BytesIO


def extract_date_from_image_path(image_path: str) -> Optional[datetime]:
    """
    Extract date from image path.
    
    Image naming: {variable}_{init_time}_{lead_time}_{date}.1.png
    Example: t_z_1000_0000_12_20200101.1.png -> 2020-01-01
    """
    # Extract YYYYMMDD pattern from filename
    pattern = r'(\d{8})\.1\.png'
    match = re.search(pattern, image_path)
    if match:
        date_str = match.group(1)
        try:
            return datetime.strptime(date_str, '%Y%m%d')
        except ValueError:
            return None
    return None


def get_season(date: datetime) -> str:
    """Get season from date."""
    month = date.month
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"


def image_to_base64(image_path: str, max_size: tuple = (400, 300)) -> str:
    """Convert image to base64 string for HTML embedding."""
    try:
        img = Image.open(image_path)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        return f"data:image/png;base64,{base64.b64encode(b'Error loading image').decode()}"


def compute_statistics(df: pd.DataFrame, metric_col: str) -> Dict[str, float]:
    """Compute comprehensive statistics for a metric."""
    values = df[metric_col].dropna()
    if len(values) == 0:
        return {}
    
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "q25": float(values.quantile(0.25)),
        "q75": float(values.quantile(0.75)),
        "count": int(len(values))
    }


def create_html_report(
    results_df: pd.DataFrame,
    test_records: List[Dict[str, Any]],
    output_dir: str,
    model_name: str = "Model"
):
    """Create comprehensive HTML visualization report."""
    
    # Add season information
    results_df['date'] = None
    results_df['season'] = None
    
    for idx, row in results_df.iterrows():
        # Get first image path from test records
        sample_id = int(row['sample_id'])
        if sample_id < len(test_records):
            image_paths = test_records[sample_id]['image_paths']
            if image_paths:
                date = extract_date_from_image_path(image_paths[0])
                if date:
                    results_df.at[idx, 'date'] = date
                    results_df.at[idx, 'season'] = get_season(date)
    
    # Compute overall statistics
    metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'meteor']
    overall_stats = {}
    for metric in metrics:
        overall_stats[metric] = compute_statistics(results_df, metric)
    
    # Compute season statistics
    season_stats = {}
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        season_df = results_df[results_df['season'] == season]
        if len(season_df) > 0:
            season_stats[season] = {}
            for metric in metrics:
                season_stats[season][metric] = compute_statistics(season_df, metric)
    
    # Create HTML content
    html_content = generate_html(
        results_df,
        test_records,
        overall_stats,
        season_stats,
        model_name
    )
    
    # Save HTML file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    html_file = output_path / "evaluation_report.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✓ Created visualization report: {html_file}")
    print(f"  Open in browser to view: file://{html_file.absolute()}")
    
    # Also save statistics as JSON
    stats_file = output_path / "statistics.json"
    stats_data = {
        "overall": overall_stats,
        "by_season": season_stats,
        "total_samples": len(results_df)
    }
    with open(stats_file, 'w') as f:
        json.dump(stats_data, f, indent=2)
    print(f"✓ Saved statistics to: {stats_file}")


def generate_html(
    results_df: pd.DataFrame,
    test_records: List[Dict[str, Any]],
    overall_stats: Dict[str, Dict[str, float]],
    season_stats: Dict[str, Dict[str, Dict[str, float]]],
    model_name: str
) -> str:
    """Generate HTML content for the report."""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>WeatherLMM Evaluation Report - {model_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #555;
            margin-top: 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            color: white;
            margin-top: 0;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-details {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .sample-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            background-color: #fafafa;
        }}
        .sample-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .sample-id {{
            font-weight: bold;
            color: #2c3e50;
        }}
        .metrics-badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            margin: 0 5px;
            font-size: 0.85em;
        }}
        .metric-good {{
            background-color: #2ecc71;
            color: white;
        }}
        .metric-medium {{
            background-color: #f39c12;
            color: white;
        }}
        .metric-poor {{
            background-color: #e74c3c;
            color: white;
        }}
        .images-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            margin: 15px 0;
        }}
        .image-container {{
            border: 1px solid #ddd;
            border-radius: 4px;
            overflow: hidden;
        }}
        .image-container img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .image-label {{
            padding: 5px;
            background-color: #34495e;
            color: white;
            font-size: 0.8em;
            text-align: center;
        }}
        .text-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 15px 0;
        }}
        .text-box {{
            padding: 15px;
            border-radius: 4px;
        }}
        .reference-box {{
            background-color: #e8f5e9;
            border-left: 4px solid #4caf50;
        }}
        .prediction-box {{
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
        }}
        .text-box h4 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .text-content {{
            white-space: pre-wrap;
            line-height: 1.6;
        }}
        .season-section {{
            margin: 30px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
        }}
        .tabs {{
            display: flex;
            border-bottom: 2px solid #ddd;
            margin-bottom: 20px;
        }}
        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 1em;
            color: #666;
        }}
        .tab.active {{
            color: #3498db;
            border-bottom: 2px solid #3498db;
            font-weight: bold;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
    </style>
    <script>
        function showTab(tabName) {{
            // Hide all tab contents
            var contents = document.getElementsByClassName('tab-content');
            for (var i = 0; i < contents.length; i++) {{
                contents[i].classList.remove('active');
            }}
            
            // Remove active class from all tabs
            var tabs = document.getElementsByClassName('tab');
            for (var i = 0; i < tabs.length; i++) {{
                tabs[i].classList.remove('active');
            }}
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }}
    </script>
</head>
<body>
    <div class="container">
        <h1>WeatherLMM Evaluation Report</h1>
        <p><strong>Model:</strong> {model_name}</p>
        <p><strong>Total Samples:</strong> {len(results_df)}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('overview')">Overview</button>
            <button class="tab" onclick="showTab('samples')">Samples</button>
            <button class="tab" onclick="showTab('seasons')">Season Analysis</button>
        </div>
        
        <!-- Overview Tab -->
        <div id="overview" class="tab-content active">
            <h2>Overall Statistics</h2>
            <div class="stats-grid">
"""
    
    # Add metric cards
    for metric in ['bleu', 'rouge1', 'rouge2', 'rougeL', 'meteor']:
        stats = overall_stats.get(metric, {})
        mean = stats.get('mean', 0.0)
        html += f"""
                <div class="stat-card">
                    <h3>{metric.upper()}</h3>
                    <div class="stat-value">{mean:.4f}</div>
                    <div class="stat-details">
                        Median: {stats.get('median', 0.0):.4f} | 
                        Std: {stats.get('std', 0.0):.4f}<br>
                        Min: {stats.get('min', 0.0):.4f} | 
                        Max: {stats.get('max', 0.0):.4f}<br>
                        Q25: {stats.get('q25', 0.0):.4f} | 
                        Q75: {stats.get('q75', 0.0):.4f}
                    </div>
                </div>
"""
    
    html += """
            </div>
            
            <h2>Detailed Statistics Table</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Q25</th>
                    <th>Q75</th>
                </tr>
"""
    
    for metric in ['bleu', 'rouge1', 'rouge2', 'rougeL', 'meteor']:
        stats = overall_stats.get(metric, {})
        html += f"""
                <tr>
                    <td><strong>{metric.upper()}</strong></td>
                    <td>{stats.get('mean', 0.0):.4f}</td>
                    <td>{stats.get('median', 0.0):.4f}</td>
                    <td>{stats.get('std', 0.0):.4f}</td>
                    <td>{stats.get('min', 0.0):.4f}</td>
                    <td>{stats.get('max', 0.0):.4f}</td>
                    <td>{stats.get('q25', 0.0):.4f}</td>
                    <td>{stats.get('q75', 0.0):.4f}</td>
                </tr>
"""
    
    html += """
            </table>
        </div>
        
        <!-- Samples Tab -->
        <div id="samples" class="tab-content">
            <h2>Sample Predictions</h2>
"""
    
    # Add sample cards (show first 20)
    for idx, row in results_df.head(20).iterrows():
        sample_id = int(row['sample_id'])
        if sample_id < len(test_records):
            image_paths = test_records[sample_id]['image_paths']
            reference = row['reference']
            prediction = row['prediction']
            
            # Get metric badges
            bleu = row['bleu']
            rouge1 = row['rouge1']
            rougeL = row['rougeL']
            
            html += f"""
            <div class="sample-card">
                <div class="sample-header">
                    <span class="sample-id">Sample #{sample_id}</span>
                    <div>
                        <span class="metrics-badge {'metric-good' if bleu > 0.2 else 'metric-poor'}">BLEU: {bleu:.3f}</span>
                        <span class="metrics-badge {'metric-good' if rouge1 > 0.4 else 'metric-poor'}">ROUGE-1: {rouge1:.3f}</span>
                        <span class="metrics-badge {'metric-good' if rougeL > 0.3 else 'metric-poor'}">ROUGE-L: {rougeL:.3f}</span>
                    </div>
                </div>
                
                <h3>Weather Charts (12 images)</h3>
                <div class="images-grid">
"""
            
            # Add images (show first 4 as thumbnails)
            image_labels = [
                'T+Z 1000hPa', 'T+Z 200hPa', 'T+Z 500hPa', 'T+Z 700hPa',
                'T+Z 850hPa', 'T2m+Wind', 'Thickness+MSLP',
                'UV+RH 1000hPa', 'UV+RH 200hPa', 'UV+RH 500hPa',
                'UV+RH 700hPa', 'UV+RH 850hPa'
            ]
            
            for i, img_path in enumerate(image_paths[:12]):
                img_b64 = image_to_base64(img_path)
                label = image_labels[i] if i < len(image_labels) else f'Image {i+1}'
                html += f"""
                    <div class="image-container">
                        <img src="{img_b64}" alt="{label}">
                        <div class="image-label">{label}</div>
                    </div>
"""
            
            html += """
                </div>
                
                <h3>Text Comparison</h3>
                <div class="text-comparison">
                    <div class="text-box reference-box">
                        <h4>Reference (Ground Truth)</h4>
                        <div class="text-content">"""
            html += reference.replace('<', '&lt;').replace('>', '&gt;')
            html += """</div>
                    </div>
                    <div class="text-box prediction-box">
                        <h4>Prediction (Model Output)</h4>
                        <div class="text-content">"""
            html += prediction.replace('<', '&lt;').replace('>', '&gt;')
            html += """</div>
                    </div>
                </div>
            </div>
"""
    
    html += """
        </div>
        
        <!-- Season Analysis Tab -->
        <div id="seasons" class="tab-content">
            <h2>Season-by-Season Analysis</h2>
"""
    
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        if season in season_stats:
            html += f"""
            <div class="season-section">
                <h3>{season}</h3>
                <p><strong>Samples:</strong> {len(results_df[results_df['season'] == season])}</p>
                <div class="stats-grid">
"""
            for metric in ['bleu', 'rouge1', 'rouge2', 'rougeL', 'meteor']:
                stats = season_stats[season].get(metric, {})
                mean = stats.get('mean', 0.0)
                html += f"""
                    <div class="stat-card">
                        <h3>{metric.upper()}</h3>
                        <div class="stat-value">{mean:.4f}</div>
                        <div class="stat-details">
                            Median: {stats.get('median', 0.0):.4f} | 
                            Std: {stats.get('std', 0.0):.4f}
                        </div>
                    </div>
"""
            html += """
                </div>
            </div>
"""
    
    html += """
        </div>
    </div>
</body>
</html>
"""
    
    return html


def load_test_manifest(csv_path: str) -> List[Dict[str, Any]]:
    """Load test manifest CSV."""
    df = pd.read_csv(csv_path)
    
    records = []
    for _, row in df.iterrows():
        if "image_paths" in df.columns:
            image_paths_str = row["image_paths"]
            image_paths = [path.strip() for path in image_paths_str.split(';')]
            target_text = row["target_text"]
            records.append({
                "image_paths": image_paths,
                "target_text": target_text
            })
    
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Visualize evaluation results with images and statistics"
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        required=True,
        help="Path to detailed_results.csv from evaluate.py"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        required=True,
        help="Path to test manifest CSV (for image paths)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_visualization",
        help="Output directory for visualization files"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="WeatherLMM",
        help="Model name for report title"
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading evaluation results from: {args.results_csv}")
    results_df = pd.read_csv(args.results_csv)
    print(f"Loaded {len(results_df)} results")
    
    # Load test manifest for image paths
    print(f"Loading test manifest from: {args.test_csv}")
    test_records = load_test_manifest(args.test_csv)
    print(f"Loaded {len(test_records)} test records")
    
    # Create visualization
    print("\nGenerating visualization report...")
    create_html_report(
        results_df,
        test_records,
        args.output_dir,
        args.model_name
    )
    
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()

"""
Evaluation script for WeatherLMM fine-tuned models.

This script evaluates a fine-tuned Qwen2.5-VL model on a test set by:
1. Loading the model (base or LoRA fine-tuned)
2. Running inference on test manifest
3. Computing evaluation metrics (BLEU, ROUGE, METEOR, etc.)
4. Generating comparison reports

Usage:
    # Evaluate LoRA fine-tuned model
    python evaluate.py \
        --test_csv ./manifests/manifest_test.csv \
        --model_path ./checkpoints/weather_lora \
        --output_dir ./evaluation_results \
        --batch_size 1

    # Evaluate base model (no fine-tuning)
    python evaluate.py \
        --test_csv ./manifests/manifest_test.csv \
        --model_name Qwen/Qwen2.5-VL-3B-Instruct \
        --output_dir ./evaluation_results \
        --batch_size 1
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. Cannot load LoRA models.")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    NLTK_AVAILABLE = True
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. BLEU and METEOR metrics will be skipped.")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not available. ROUGE metrics will be skipped.")


class WeatherForecastEvaluator:
    """Evaluator for weather forecast generation models."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        model_path: Optional[str] = None,
        device: str = "cuda",
        max_new_tokens: int = 2048
    ):
        """
        Initialize the evaluator.

        Args:
            model_name: Base model name (if model_path is None, uses this)
            model_path: Path to fine-tuned model directory (LoRA checkpoint)
            device: Device to run inference on
            max_new_tokens: Maximum tokens to generate
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens

        print(f"Loading model: {model_name}")
        if model_path:
            print(f"  Fine-tuned checkpoint: {model_path}")

        # Load base model
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load LoRA adapter if provided
        if model_path and PEFT_AVAILABLE:
            print(f"Loading LoRA adapter from {model_path}...")
            self.model = PeftModel.from_pretrained(self.model, model_path)
            print("✓ LoRA adapter loaded")
        elif model_path and not PEFT_AVAILABLE:
            raise ImportError("PEFT required to load LoRA models. Install with: pip install peft")

        self.model.eval()
        print(f"Model loaded on {self.device}")

    def generate_forecast(self, images: List[Image.Image]) -> str:
        """
        Generate forecast from multiple weather chart images.

        Args:
            images: List of 12 PIL Images (weather charts)

        Returns:
            Generated forecast text
        """
        # Use the same prompt as training
        prompt = (
            "You are a weather forecasting assistant. "
            "Given these 12 forecast maps (temperature and geopotential at various pressure levels,"
            "temperature and wind, thickness and mean sea level pressure,"
            "and lastly wind and relative humidity at various pressure levels),"
            "describe the expected weather in detail."
        )

        # Build conversation format with actual images (matching training format)
        user_message = {
            "role": "user",
            "content": [
                {"type": "image", "image": img} for img in images
            ] + [{"type": "text", "text": prompt}]
        }

        messages = [user_message]

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
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Deterministic
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

        return output_text.strip()


def load_test_manifest(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load test manifest CSV.

    Args:
        csv_path: Path to manifest CSV

    Returns:
        List of records with image_paths and target_text
    """
    df = pd.read_csv(csv_path)
    
    records = []
    for _, row in df.iterrows():
        if "image_paths" in df.columns:
            # Semicolon-separated format
            image_paths_str = row["image_paths"]
            image_paths = [path.strip() for path in image_paths_str.split(';')]
            target_text = row["target_text"]
        else:
            raise ValueError("Manifest must contain 'image_paths' and 'target_text' columns")
        
        records.append({
            "image_paths": image_paths,
            "target_text": target_text
        })
    
    return records


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute BLEU score."""
    if not NLTK_AVAILABLE:
        return 0.0
    
    try:
        from nltk.tokenize import word_tokenize
        ref_tokens = word_tokenize(reference.lower())
        hyp_tokens = word_tokenize(hypothesis.lower())
        
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
        return score
    except Exception:
        return 0.0


def compute_rouge(reference: str, hypothesis: str) -> Dict[str, float]:
    """Compute ROUGE scores."""
    if not ROUGE_AVAILABLE:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure,
        }
    except Exception:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def compute_meteor(reference: str, hypothesis: str) -> float:
    """Compute METEOR score."""
    if not NLTK_AVAILABLE:
        return 0.0
    
    try:
        from nltk.tokenize import word_tokenize
        ref_tokens = word_tokenize(reference.lower())
        hyp_tokens = word_tokenize(hypothesis.lower())
        
        score = meteor_score([ref_tokens], hyp_tokens)
        return score
    except Exception:
        return 0.0


def evaluate(
    evaluator: WeatherForecastEvaluator,
    test_records: List[Dict[str, Any]],
    batch_size: int = 1,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate model on test set.

    Args:
        evaluator: WeatherForecastEvaluator instance
        test_records: List of test records
        batch_size: Batch size for inference (currently only 1 supported)
        max_samples: Maximum number of samples to evaluate (None = all)

    Returns:
        Dictionary with metrics and results
    """
    if max_samples:
        test_records = test_records[:max_samples]
    
    print(f"\nEvaluating on {len(test_records)} samples...")
    
    predictions = []
    references = []
    results = []
    
    for i, record in enumerate(tqdm(test_records, desc="Generating forecasts")):
        # Load images
        image_paths = record["image_paths"]
        images = []
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
            images.append(Image.open(img_path).convert("RGB"))
        
        if len(images) != 12:
            print(f"Warning: Expected 12 images, got {len(images)}. Skipping.")
            continue
        
        # Generate prediction
        try:
            prediction = evaluator.generate_forecast(images)
        except Exception as e:
            print(f"Error generating forecast for sample {i}: {e}")
            prediction = ""
        
        reference = record["target_text"]
        
        predictions.append(prediction)
        references.append(reference)
        
        # Compute metrics for this sample
        bleu = compute_bleu(reference, prediction)
        rouge = compute_rouge(reference, prediction)
        meteor = compute_meteor(reference, prediction)
        
        results.append({
            "sample_id": i,
            "prediction": prediction,
            "reference": reference,
            "bleu": bleu,
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
            "meteor": meteor,
        })
    
    # Compute aggregate metrics
    avg_bleu = sum(r["bleu"] for r in results) / len(results) if results else 0.0
    avg_rouge1 = sum(r["rouge1"] for r in results) / len(results) if results else 0.0
    avg_rouge2 = sum(r["rouge2"] for r in results) / len(results) if results else 0.0
    avg_rougeL = sum(r["rougeL"] for r in results) / len(results) if results else 0.0
    avg_meteor = sum(r["meteor"] for r in results) / len(results) if results else 0.0
    
    metrics = {
        "num_samples": len(results),
        "bleu": avg_bleu,
        "rouge1": avg_rouge1,
        "rouge2": avg_rouge2,
        "rougeL": avg_rougeL,
        "meteor": avg_meteor,
    }
    
    return {
        "metrics": metrics,
        "results": results,
        "predictions": predictions,
        "references": references,
    }


def save_results(
    evaluation_results: Dict[str, Any],
    output_dir: str,
    model_name: str
):
    """Save evaluation results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save metrics summary
    metrics_file = output_path / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(evaluation_results["metrics"], f, indent=2)
    print(f"\n✓ Saved metrics to: {metrics_file}")
    
    # Save detailed results CSV
    results_file = output_path / "detailed_results.csv"
    df = pd.DataFrame(evaluation_results["results"])
    df.to_csv(results_file, index=False)
    print(f"✓ Saved detailed results to: {results_file}")
    
    # Save predictions and references side-by-side
    comparison_file = output_path / "predictions_vs_references.csv"
    comparison_data = []
    for r in evaluation_results["results"]:
        comparison_data.append({
            "sample_id": r["sample_id"],
            "reference": r["reference"],
            "prediction": r["prediction"],
            "bleu": r["bleu"],
            "rouge1": r["rouge1"],
            "rouge2": r["rouge2"],
            "rougeL": r["rougeL"],
            "meteor": r["meteor"],
        })
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison.to_csv(comparison_file, index=False)
    print(f"✓ Saved comparison to: {comparison_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Number of samples: {evaluation_results['metrics']['num_samples']}")
    print(f"\nMetrics:")
    print(f"  BLEU:  {evaluation_results['metrics']['bleu']:.4f}")
    print(f"  ROUGE-1: {evaluation_results['metrics']['rouge1']:.4f}")
    print(f"  ROUGE-2: {evaluation_results['metrics']['rouge2']:.4f}")
    print(f"  ROUGE-L: {evaluation_results['metrics']['rougeL']:.4f}")
    print(f"  METEOR: {evaluation_results['metrics']['meteor']:.4f}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate WeatherLMM fine-tuned models"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        required=True,
        help="Path to test manifest CSV"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Base model name (if --model_path not provided)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned LoRA checkpoint directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (currently only 1 supported)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for quick testing)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # Load test manifest
    print(f"Loading test manifest from: {args.test_csv}")
    test_records = load_test_manifest(args.test_csv)
    print(f"Loaded {len(test_records)} test samples")
    
    # Initialize evaluator
    evaluator = WeatherForecastEvaluator(
        model_name=args.model_name,
        model_path=args.model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens
    )
    
    # Run evaluation
    evaluation_results = evaluate(
        evaluator,
        test_records,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # Save results
    model_identifier = args.model_path if args.model_path else args.model_name
    save_results(
        evaluation_results,
        args.output_dir,
        model_identifier
    )


if __name__ == "__main__":
    main()

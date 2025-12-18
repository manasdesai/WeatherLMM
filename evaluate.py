"""
Evaluation script for WeatherLMM fine-tuned models.

This script evaluates a fine-tuned Qwen2.5-VL model on a test set by:
1. Loading the model (base or LoRA fine-tuned)
2. Running inference on test manifest
3. Computing evaluation metrics (BLEU, ROUGE, METEOR, etc.)
4. Generating comparison reports

Usage:
    # Evaluate LoRA fine-tuned model (greedy decoding - deterministic)
    # IMPORTANT: Use --image_size 448 to match training (much faster!)
    python evaluate.py \
        --test_csv ./manifests/manifest_test.csv \
        --model_path ./checkpoints/weather_lora \
        --output_dir ./evaluation_results \
        --image_size 448 \
        --batch_size 1

    # Evaluate with sampling (for more diverse outputs)
    python evaluate.py \
        --test_csv ./manifests/manifest_test.csv \
        --model_path ./checkpoints/weather_lora \
        --output_dir ./evaluation_results \
        --image_size 448 \
        --do_sample \
        --temperature 0.7 \
        --top_p 0.9

    # Evaluate base model (no fine-tuning)
    python evaluate.py \
        --test_csv ./manifests/manifest_test.csv \
        --model_name Qwen/Qwen2.5-VL-3B-Instruct \
        --output_dir ./evaluation_results \
        --image_size 448 \
        --batch_size 1
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import csv
import random
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
        max_new_tokens: int = 1500,
        image_size: Optional[int] = 448
    ):
        """
        Initialize the evaluator.

        Args:
            model_name: Base model name (if model_path is None, uses this)
            model_path: Path to fine-tuned model directory (LoRA checkpoint)
            device: Device to run inference on
            max_new_tokens: Maximum tokens to generate
            image_size: Optional square resize for all input images (e.g., 448). 
                       Should match training image_size for consistency. 
                       If None, original image resolution is used.
        """
        # Check CUDA availability and set device
        cuda_available = torch.cuda.is_available()
        if device == "cuda" and not cuda_available:
            print(f"WARNING: CUDA requested but not available. Falling back to CPU.")
            print(f"   This will be significantly slower. Install CUDA-enabled PyTorch for GPU acceleration.")
            self.device = "cpu"
        else:
            self.device = device if cuda_available else "cpu"
        
        self.max_new_tokens = max_new_tokens
        self.image_size = image_size
        
        if image_size:
            print(f"Image resizing enabled: {image_size}x{image_size} (matching training)")

        # Print GPU information
        if cuda_available:
            num_gpus = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
            print(f"CUDA available: {num_gpus} GPU(s) detected")
            print(f"  Using GPU {current_device}: {gpu_name} ({gpu_memory:.2f} GB)")
        else:
            print(f"Running on CPU (no GPU detected)")

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
            print("LoRA adapter loaded")
        elif model_path and not PEFT_AVAILABLE:
            raise ImportError("PEFT required to load LoRA models. Install with: pip install peft")

        self.model.eval()
        
        # Optimize model for inference if torch.compile is available (PyTorch 2.0+)
        # Note: torch.compile may not work well with LoRA/PEFT models, so we skip it for now
        # If you want to try it, uncomment below, but test carefully
        # try:
        #     if hasattr(torch, 'compile') and self.device == "cuda":
        #         print("Optimizing model with torch.compile for faster inference...")
        #         self.model = torch.compile(self.model, mode="reduce-overhead")
        #         print("  Model compiled successfully")
        # except Exception as e:
        #     print(f"  Could not compile model: {e}")
        
        # Enable better memory efficiency
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster matmuls on Ampere+
        
        # Verify model device placement
        try:
            model_device = next(self.model.parameters()).device
            if model_device.type == "cuda":
                print(f"Model verified on GPU: {model_device}")
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(model_device) / 1024**3
                    reserved = torch.cuda.memory_reserved(model_device) / 1024**3
                    print(f"  GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
            else:
                print(f"WARNING: Model is on {model_device}, not GPU! This will be slow.")
        except Exception as e:
            print(f"WARNING: Could not verify model device: {e}")
            print(f"   Assuming model is on {self.device}")

    def generate_forecast(
        self, 
        images: List[Image.Image],
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0
    ) -> str:
        """
        Generate forecast from multiple weather chart images.

        Args:
            images: List of 12 PIL Images (weather charts)
            do_sample: If True, use sampling instead of greedy decoding
            temperature: Sampling temperature (only used if do_sample=True)
            top_p: Nucleus sampling parameter (only used if do_sample=True)

        Returns:
            Generated forecast text
        
        Note: Processing 12 images through the vision encoder is computationally
        expensive. Expected time per sample: 60-120 seconds depending on GPU and
        generation length. The timing breakdown will show where time is spent.
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
        import time
        start_time = time.time()
        
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
        
        prep_time = time.time() - start_time

        # Move inputs to the same device as the model
        try:
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v 
                      for k, v in inputs.items()}
        except Exception:
            # Fallback to self.device if we can't determine model device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in inputs.items()}

        # Generate forecast with optimizations
        gen_start = time.time()
        # Use inference_mode() instead of no_grad() for faster inference
        with torch.inference_mode():
            # Get tokenizer for stopping criteria
            tokenizer = self.processor.tokenizer
            
            generate_kwargs = {
                **inputs,
                "max_new_tokens": self.max_new_tokens,
                "do_sample": do_sample,
                "use_cache": True,  # Enable KV cache for faster generation
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_p"] = top_p
            else:
                # For greedy decoding, ensure we don't pass sampling parameters
                # Some models don't accept temperature when do_sample=False
                pass
            
            generated_ids = self.model.generate(**generate_kwargs)
        
        gen_time = time.time() - gen_start

        # Decode output
        # inputs is a dict, so access input_ids as a key
        input_ids = inputs["input_ids"]
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        
        # Check actual generation length (for debugging)
        actual_tokens = len(generated_ids_trimmed[0])
        if not hasattr(self, '_token_stats'):
            self._token_stats = []
        self._token_stats.append(actual_tokens)
        if len(self._token_stats) == 1:
            print(f"  Generated {actual_tokens} tokens (max allowed: {self.max_new_tokens})")

        decode_start = time.time()
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        decode_time = time.time() - decode_start
        
        total_time = time.time() - start_time
        # Only print timing for first sample to identify bottlenecks
        if not hasattr(self, '_timing_printed'):
            print(f"\nTiming breakdown (first sample):")
            print(f"     - Image prep: {prep_time:.2f}s")
            print(f"     - Generation: {gen_time:.2f}s")
            print(f"     - Decoding: {decode_time:.2f}s")
            print(f"     - Total: {total_time:.2f}s")
            
            self._timing_printed = True
        
        # Clear GPU cache periodically to prevent memory buildup
        if self.device == "cuda" and hasattr(self, '_sample_count'):
            self._sample_count += 1
            if self._sample_count % 10 == 0:
                torch.cuda.empty_cache()
        elif self.device == "cuda":
            self._sample_count = 1

        return output_text.strip()


def load_test_manifest(csv_path: str, shuffle: bool = False, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load test manifest CSV and filter out samples with missing images.

    Args:
        csv_path: Path to manifest CSV
        shuffle: If True, shuffle the records after filtering
        seed: Random seed for shuffling (for reproducibility)

    Returns:
        List of records with image_paths and target_text (only valid samples)
    """
    df = pd.read_csv(csv_path)
    
    records = []
    filtered_count = 0
    
    for idx, row in df.iterrows():
        if "image_paths" in df.columns:
            # Semicolon-separated format
            image_paths_str = row["image_paths"]
            if pd.isna(image_paths_str) or not image_paths_str:
                filtered_count += 1
                continue
            
            image_paths = [path.strip() for path in str(image_paths_str).split(';') if path.strip()]
            
            # Validate we have exactly 12 images
            if len(image_paths) != 12:
                filtered_count += 1
                continue
            
            # Check all image paths exist
            all_exist = True
            for img_path in image_paths:
                if pd.isna(img_path) or not img_path or not os.path.exists(str(img_path)):
                    all_exist = False
                    break
            
            if not all_exist:
                filtered_count += 1
                continue
            
            # Check target_text is valid
            target_text = row["target_text"]
            if pd.isna(target_text) or not str(target_text).strip():
                filtered_count += 1
                continue
        else:
            raise ValueError("Manifest must contain 'image_paths' and 'target_text' columns")
        
        records.append({
            "image_paths": image_paths,
            "target_text": target_text
        })
    
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} samples with missing images or invalid data (from {len(df)} total).")
        print(f"Remaining: {len(records)} valid samples.")
    
    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(records)
        print(f"Shuffled evaluation samples (seed={seed}).")
    
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
        scores = scorer.score(reference.lower(), hypothesis.lower())
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
    max_samples: Optional[int] = None,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0
) -> Dict[str, Any]:
    """
    Evaluate model on test set.

    Args:
        evaluator: WeatherForecastEvaluator instance
        test_records: List of test records
        batch_size: Batch size for inference (currently only 1 supported)
        max_samples: Maximum number of samples to evaluate (None = all)
        do_sample: If True, use sampling instead of greedy decoding
        temperature: Sampling temperature (only used if do_sample=True)
        top_p: Nucleus sampling parameter (only used if do_sample=True)

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
        # Load images (should all exist since we filtered during manifest loading)
        image_paths = record["image_paths"]
        images = []
        try:
            for img_path in image_paths:
                # Double-check existence (in case files were deleted after loading)
                if not os.path.exists(img_path):
                    print(f"Warning: Image file was deleted after manifest loading: {img_path}. Skipping sample {i}.")
                    break
                img = Image.open(img_path).convert("RGB")
                # Resize image if image_size is specified (should match training)
                if evaluator.image_size is not None and evaluator.image_size > 0:
                    img = img.resize((evaluator.image_size, evaluator.image_size), Image.BICUBIC)
                images.append(img)
        except Exception as e:
            print(f"Error loading images for sample {i}: {e}. Skipping.")
            continue
        
        if len(images) != 12:
            print(f"Warning: Expected 12 images, got {len(images)} for sample {i}. Skipping.")
            continue
        
        # Generate prediction
        try:
            prediction = evaluator.generate_forecast(
                images,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p
            )
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
        default=1500,
        help="Maximum tokens to generate. Analysis of 16k+ forecasts: avg ~539 tokens, 99th percentile ~1292 tokens, max ~2009 tokens. Default 1500 covers 99%+ of forecasts while being ~25% faster than 2048."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=448,
        help="Square resize for all input images (e.g., 448). Should match training image_size for consistency. Default: 448"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda or cpu)"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling instead of greedy decoding (enables temperature and top_p)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (only used if --do_sample is set). Higher = more random. Default: 0.7"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter (only used if --do_sample is set). Default: 0.9"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle evaluation samples before processing (for randomized evaluation order)"
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Random seed for shuffling (for reproducibility). Default: 42"
    )
    
    args = parser.parse_args()
    
    # Load test manifest (filters out samples with missing images)
    print(f"Loading test manifest from: {args.test_csv}")
    test_records = load_test_manifest(
        args.test_csv,
        shuffle=args.shuffle,
        seed=args.shuffle_seed
    )
    print(f"Loaded {len(test_records)} valid test samples")
    
    # Initialize evaluator
    evaluator = WeatherForecastEvaluator(
        model_name=args.model_name,
        model_path=args.model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        image_size=args.image_size
    )
    
    # Run evaluation
    evaluation_results = evaluate(
        evaluator,
        test_records,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p
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

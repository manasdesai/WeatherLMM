import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
import transformers

from peft import LoraConfig, get_peft_model

# #region agent log
log_path = "/Users/dcalhoun/Desktop/Courses/Fall 2025/CMSC 723/Project/WeatherLMM/.cursor/debug.log"

def log_memory(location: str, message: str, hypothesis_id: str = "MEM"):
    """Log GPU memory usage."""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": hypothesis_id,
                    "location": location,
                    "message": message,
                    "data": {
                        "allocated_gb": round(allocated, 2),
                        "reserved_gb": round(reserved, 2),
                        "max_allocated_gb": round(max_allocated, 2),
                        "cuda_available": True
                    },
                    "timestamp": __import__('time').time() * 1000
                }) + "\n")
        else:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": hypothesis_id,
                    "location": location,
                    "message": message,
                    "data": {"cuda_available": False},
                    "timestamp": __import__('time').time() * 1000
                }) + "\n")
    except Exception as e:
        pass
# #endregion

# #region agent log
import json
import os
log_path = "/Users/dcalhoun/Desktop/Courses/Fall 2025/CMSC 723/Project/WeatherLMM/.cursor/debug.log"
try:
    with open(log_path, 'a') as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "LoRA_Training.py:19", "message": "Transformers version check", "data": {"transformers_version": transformers.__version__}, "timestamp": __import__('time').time() * 1000}) + "\n")
except Exception as e:
    pass
# #endregion

#loading our dataset from our manifest file.
class ImageTextDataset(Dataset):
    def __init__(self, csv_path:str, max_samples: int = None):
        """
        Initialize dataset from CSV.
        
        Args:
            csv_path: Path to CSV manifest file
            max_samples: Maximum number of samples to load (None = all samples). Useful for quick development.
        """
        self.df = pd.read_csv(csv_path)
        
        # Limit dataset size if specified (for quick development)
        if max_samples is not None and max_samples > 0:
            original_size = len(self.df)
            self.df = self.df.head(max_samples)
            print(f"Limited dataset to {len(self.df)} samples (from {original_size} total)")
        
        # Support both formats:
        # 1. Semicolon-separated image_paths column (from create_full_image_manifest.py)
        # 2. Individual columns for each image path
        if "image_paths" in self.df.columns:
            # Format: semicolon-separated paths
            self.use_semicolon_format = True
            if "target_text" not in self.df.columns:
                raise ValueError("CSV must contain 'target_text' column when using 'image_paths' format")
        else:
            # Format: individual columns
            self.use_semicolon_format = False
            self.image_columns = [
                "press_level_200_hPA_path", "press_level_500_hPA_path", "press_level_700_hPA_path",
                "press_level_850_hPA_path", "press_level_1000_hPA_path", "2m_temp_10m_wind_path",
                "500_1000_hPA_Thickness_mean_sealevel_pressure_path", "wind_relative_humidity_200_hPA_path", "wind_relative_humidity_500_hPA_path",
                "wind_relative_humidity_700_hPA_path", "wind_relative_humidity_850_hPA_path", "wind_relative_humidity_1000_hPA_path"
            ] 
            missing = [col for col in self.image_columns if col not in self.df.columns]
            if missing:
                raise ValueError(f"Missing columns in CSV: {missing}")
            if "text" not in self.df.columns:
                raise ValueError("CSV must contain 'text' column when using individual column format")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx:int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        our_weather_images = []
        
        if self.use_semicolon_format:
            # Parse semicolon-separated image paths
            image_paths_str = row["image_paths"]
            if pd.isna(image_paths_str) or not image_paths_str:
                raise ValueError(f"image_paths is empty or NaN in row {idx}")
            image_paths = [path.strip() for path in str(image_paths_str).split(';') if path.strip()]
            if len(image_paths) != 12:
                raise ValueError(f"Expected 12 image paths, got {len(image_paths)} in row {idx}")
            
            for image_path in image_paths:
                if pd.isna(image_path) or not image_path or not os.path.exists(str(image_path)):
                    raise FileNotFoundError(f"Image not found or path is empty: {image_path}")
                try:
                    our_weather_images.append(Image.open(str(image_path)).convert("RGB"))
                except Exception as e:
                    raise RuntimeError(f"Failed to load image {image_path}: {e}")
            
            text = row["target_text"]
            if pd.isna(text):
                raise ValueError(f"Text is NaN in row {idx}")
        else:
            # Use individual columns
            for col in self.image_columns:
                image_path = row[col]
                if pd.isna(image_path) or not image_path or not os.path.exists(str(image_path)):
                    raise FileNotFoundError(f"Image not found or path is empty: {image_path} (column: {col})")
                try:
                    our_weather_images.append(Image.open(str(image_path)).convert("RGB"))
                except Exception as e:
                    raise RuntimeError(f"Failed to load image {image_path} (column: {col}): {e}")
            text = row["text"]
            if pd.isna(text):
                raise ValueError(f"Text is NaN in row {idx}")

        return {"image": our_weather_images, "text": text}
    
# Defining our datacollector along with masking.
@dataclass
class WeatherDataCollectorImageText:
    
    processor: AutoProcessor
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        weather_images_batch = [item["image"] for item in batch]
        text = [item["text"] for item in batch]
        
        # #region agent log
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                with open(log_path, 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "MEM_COLLATOR",
                        "location": "LoRA_Training.py:data_collator",
                        "message": "Memory in data collator",
                        "data": {"allocated_gb": round(allocated, 2), "batch_size": len(batch)},
                        "timestamp": __import__('time').time() * 1000
                    }) + "\n")
        except:
            pass
        # #endregion
        
        full_text = []
        user_only_messages = []

        for t in text:
            user_message = {
                "role": "user",
                "content": [
                    #Adding our 12 images as context for the model.
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            "You are a weather forecasting assistant. "
                            "Given these 12 forecast maps (temperature and geopotential at various pressure levels,"
                            "temperature and wind, thickness and mean sea level pressure,"
                            "and lastely wind and relative humidity at various pressure levels),"
                            "describe the expected weather in detail."
                            
                            # Dean's suggested prompt:
                            
                            # "You are a skilled weather forecasting system operating on behalf of the National
                            # Weather Service’s Weather Prediction Center. Given this set of numerical weather prediction
                            # model output images for a +12 hour forecast, produce a short-range synoptic-scale
                            # weather forecast for the continental United States over the next 1 to 2 days.
                            # Your forecast will be relied upon by millions across the country, so it is critical
                            # to be careful and accurate. Think deeply about each weather variable and their
                            # relationships, recalling principles of quasi-geostrophic meteorology,
                            # to ensure your forecast is physically consistent before generating your final answer.”
                        ),
                    },
                ],
            }

            full_messages = [
                user_message,
                {
                    "role": "assistant",
                    "content": t,
                },
            ]
            full_text.append(full_messages)
            user_only_messages.append([user_message])

        #This is for our user only prompts where we're getting our prompt length in tokens.
        prompt_lengths = []
        for user_msg in user_only_messages:
            user_text = self.processor.apply_chat_template(
                user_msg,
                tokenize=False,
                add_generation_prompt=False,
            )
            user_tok = self.processor.tokenizer(
                user_text,
                add_special_tokens=True,
                padding=False,
            )
            prompt_lengths.append(len(user_tok["input_ids"]))
        
        #This is the full message with everything included, the user + assistant.
        full_messages = self.processor.apply_chat_template(
            full_text,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        #flattening the images for the batch processor.
        #batch size x 12 images for each.
        
        all_weather_images = []
        for images_per_sample in weather_images_batch:
            all_weather_images.extend(images_per_sample)
        
        model_inputs = self.processor(
            text=full_messages,
            images=all_weather_images, #passing all of the images in the batch flattened.
            padding=True,
            #return PyTorch tensors
            return_tensors="pt",
        )
        
        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()
        
        #Masking the users tokens so we the assistant tokens contribute to our loss.
        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = -100  # Masking the prompt part
            
        model_inputs["labels"] = labels
        
        # #region agent log
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                with open(log_path, 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "MEM_COLLATOR_END",
                        "location": "LoRA_Training.py:data_collator_end",
                        "message": "Memory after data collator processing",
                        "data": {"allocated_gb": round(allocated, 2)},
                        "timestamp": __import__('time').time() * 1000
                    }) + "\n")
        except:
            pass
        # #endregion
        
        return model_inputs
  
def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune Qwen2.5-VL on (image, caption) weather data."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Hugging Face model name.",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Training CSV with 'image_path' and 'caption' columns.",
    )
    parser.add_argument(
        "--eval_csv",
        type=str,
        default=None,
        help="Optional eval CSV with same columns.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum number of training samples to use (for quick development). None = use all samples.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Maximum number of evaluation samples to use (for quick development). None = use all samples.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./qwen2_5_vl_weather_lora",
        help="Where to save LoRA adapter + config.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,  # slightly higher is OK for LoRA
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=400,
        help="Save checkpoint every N steps. Must be a multiple of eval_steps when load_best_model_at_end=True.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=200,
        help="Evaluate every N steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 training if supported.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 training if supported.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If >0, limit total training steps for quick experiments.",
    )
    parser.add_argument(
        "--use_8bit_optimizer",
        action="store_true",
        help="Use 8-bit optimizer (bitsandbytes) to reduce memory. Requires: pip install bitsandbytes",
    )
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        help="Enable multi-GPU training (data parallelism). Uses all available GPUs.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for model parallelism. Options: 'auto', 'balanced', 'balanced_low_0', or custom dict. Use 'auto' for automatic distribution.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"Training CSV not found: {args.train_csv}")
    
    if args.eval_csv and not os.path.exists(args.eval_csv):
        raise FileNotFoundError(f"Evaluation CSV not found: {args.eval_csv}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load and validate datasets before model loading (faster failure)
    print("Loading training dataset...")
    train_dataset = ImageTextDataset(args.train_csv, max_samples=args.max_train_samples)
    if len(train_dataset) == 0:
        raise ValueError(f"Training dataset is empty: {args.train_csv}")
    print(f"Training samples: {len(train_dataset)}")
    if args.max_train_samples:
        print(f"  (Limited from full dataset for quick development)")
    
    eval_dataset = None
    if args.eval_csv:
        print("Loading evaluation dataset...")
        eval_dataset = ImageTextDataset(args.eval_csv, max_samples=args.max_eval_samples)
        if len(eval_dataset) == 0:
            raise ValueError(f"Evaluation dataset is empty: {args.eval_csv}")
        print(f"Evaluation samples: {len(eval_dataset)}")
        if args.max_eval_samples:
            print(f"  (Limited from full dataset for quick development)")
    
    # Validate dataset content (check first few samples)
    print("Validating dataset content...")
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        if not sample["text"] or pd.isna(sample["text"]) or len(str(sample["text"]).strip()) == 0:
            raise ValueError(f"Empty or NaN text found in training sample {i}")
        if len(sample["image"]) != 12:
            raise ValueError(f"Expected 12 images, got {len(sample['image'])} in training sample {i}")
    
    if eval_dataset:
        for i in range(min(3, len(eval_dataset))):
            sample = eval_dataset[i]
            if not sample["text"] or pd.isna(sample["text"]) or len(str(sample["text"]).strip()) == 0:
                raise ValueError(f"Empty or NaN text found in evaluation sample {i}")
            if len(sample["image"]) != 12:
                raise ValueError(f"Expected 12 images, got {len(sample['image'])} in evaluation sample {i}")
    
    print("Dataset validation passed.")
    
    # #region agent log
    log_memory("LoRA_Training.py:before_model_load", "Memory before model loading", "MEM1")
    # #endregion
    
    # Check available GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.2f} GB)")
    else:
        num_gpus = 0
        print("No CUDA GPUs available")
    
    # Determine device_map strategy
    if num_gpus > 1:
        if args.device_map == "auto":
            # Auto-distribute model across available GPUs
            device_map_strategy = "auto"
            print(f"\nUsing model parallelism: Auto-distributing model across {num_gpus} GPUs")
        elif args.device_map == "balanced":
            device_map_strategy = "balanced"
            print(f"\nUsing model parallelism: Balanced distribution across {num_gpus} GPUs")
        else:
            device_map_strategy = args.device_map
            print(f"\nUsing custom device_map: {device_map_strategy}")
    else:
        device_map_strategy = None if num_gpus == 0 else "auto"
        if num_gpus == 1:
            print(f"\nSingle GPU detected. Using device_map='auto' for optimal memory usage.")
    
    # When using device_map (model parallelism), prevent Accelerate from initializing distributed training
    # Device_map splits model across GPUs, which is incompatible with DDP (data parallelism)
    if device_map_strategy is not None and device_map_strategy != "cpu":
        # Set environment variables to prevent distributed initialization
        # os is already imported at the top of the file
        os.environ["ACCELERATE_USE_CPU"] = "false"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        # Most importantly: tell Accelerate this is NOT a distributed run
        os.environ["LOCAL_RANK"] = "-1"
        os.environ["RANK"] = "-1"
        os.environ["WORLD_SIZE"] = "1"
    
    # Load model with error handling
    print(f"Loading model: {args.model_name}")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_name,
                torch_dtype=(torch.float16 if args.fp16 else None),
                device_map=device_map_strategy,
            )
        # #region agent log
        log_memory("LoRA_Training.py:after_model_load", "Memory after model loading", "MEM2")
        # #endregion
        
        # Clear cache after model loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # #region agent log
            log_memory("LoRA_Training.py:after_cache_clear", "Memory after cache clear", "MEM3")
            # #endregion
    except Exception as e:
        raise RuntimeError(f"Failed to load model {args.model_name}: {e}")
    
    # Load processor with error handling
    try:
        processor = AutoProcessor.from_pretrained(args.model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load processor for {args.model_name}: {e}")
    
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model.config.pad_token_id = processor.tokenizer.eos_token_id
        
    
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],    
    )
    
    # Apply LoRA with validation
    try:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        # #region agent log
        log_memory("LoRA_Training.py:after_lora", "Memory after LoRA application", "MEM4")
        # #endregion
        
        # Clear cache after LoRA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        raise RuntimeError(f"Failed to apply LoRA configuration: {e}. Check that target_modules exist in the model.")
    
    data_collator = WeatherDataCollectorImageText(processor=processor)
    
    # #region agent log
    log_memory("LoRA_Training.py:before_training_args", "Memory before TrainingArguments", "MEM5")
    # #endregion
    
    # Validate training configuration
    if args.max_steps > 0 and args.num_train_epochs > 0:
        print(f"Warning: Both max_steps ({args.max_steps}) and num_train_epochs ({args.num_train_epochs}) are set.")
        print(f"max_steps will take precedence. Training will stop after {args.max_steps} steps.")
    
    # Validate and adjust save_steps/eval_steps relationship when load_best_model_at_end=True
    load_best_model = eval_dataset is not None
    save_steps = args.save_steps
    eval_steps = args.eval_steps
    
    # Determine eval_strategy and save_strategy
    eval_strategy = "steps" if eval_dataset is not None else "no"
    save_strategy = eval_strategy if load_best_model else "steps"
    
    if load_best_model and save_steps % eval_steps != 0:
        # Auto-adjust save_steps to be the next multiple of eval_steps
        adjusted_save_steps = ((save_steps // eval_steps) + 1) * eval_steps
        print(f"Warning: save_steps ({save_steps}) is not a multiple of eval_steps ({eval_steps}).")
        print(f"Auto-adjusting save_steps to {adjusted_save_steps} to satisfy load_best_model_at_end requirement.")
        save_steps = adjusted_save_steps
    
    # #region agent log
    import inspect
    try:
        with open(log_path, 'a') as f:
            sig = inspect.signature(TrainingArguments.__init__)
            params = list(sig.parameters.keys())
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "LoRA_Training.py:317", "message": "TrainingArguments parameters check", "data": {"has_evaluation_strategy": "evaluation_strategy" in params, "has_eval_strategy": "eval_strategy" in params, "all_params": params[:20]}, "timestamp": __import__('time').time() * 1000}) + "\n")
    except Exception as e:
        pass
    
    # Log distributed state before TrainingArguments
    try:
        import torch.distributed as dist
        is_distributed = dist.is_initialized() if hasattr(dist, 'is_initialized') else False
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": "LoRA_Training.py:before_training_args", "message": "Distributed state check", "data": {"num_gpus": num_gpus, "device_map_strategy": device_map_strategy if 'device_map_strategy' in locals() else None, "is_distributed_initialized": is_distributed, "torch_distributed_available": hasattr(torch.distributed, 'is_initialized')}, "timestamp": __import__('time').time() * 1000}) + "\n")
    except Exception as e:
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": "LoRA_Training.py:before_training_args", "message": "Distributed state check failed", "data": {"error": str(e)}, "timestamp": __import__('time').time() * 1000}) + "\n")
    # #endregion
    
    # Determine if we should use distributed training
    # When using device_map (model parallelism), we should NOT use DDP (data parallelism)
    # They are incompatible - device_map splits model across GPUs, DDP replicates model on each GPU
    use_ddp = False
    if num_gpus > 1 and device_map_strategy is None:
        # Only use DDP if we have multiple GPUs AND we're NOT using device_map
        use_ddp = True
        print(f"\nUsing DistributedDataParallel (DDP) for data parallelism across {num_gpus} GPUs")
    elif num_gpus > 1 and device_map_strategy is not None:
        print(f"\nUsing model parallelism (device_map={device_map_strategy}). DDP disabled (incompatible with model parallelism).")
        print("Model will be split across GPUs, but training runs in single-process mode.")
    
    # When using device_map, prevent Accelerate from initializing distributed training
    if device_map_strategy is not None and device_map_strategy != "cpu":
        # Set environment variables to prevent distributed initialization
        # os is already imported at the top of the file
        os.environ["LOCAL_RANK"] = "-1"
        os.environ["RANK"] = "-1"
        os.environ["WORLD_SIZE"] = "1"
        # #region agent log
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H4", "location": "LoRA_Training.py:env_vars_set", "message": "Set env vars to disable DDP", "data": {"LOCAL_RANK": os.environ.get("LOCAL_RANK"), "RANK": os.environ.get("RANK"), "WORLD_SIZE": os.environ.get("WORLD_SIZE")}, "timestamp": __import__('time').time() * 1000}) + "\n")
        except:
            pass
        # #endregion
    
    # #region agent log
    try:
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H2", "location": "LoRA_Training.py:ddp_decision", "message": "DDP usage decision", "data": {"use_ddp": use_ddp, "num_gpus": num_gpus, "device_map_strategy": device_map_strategy if 'device_map_strategy' in locals() else None}, "timestamp": __import__('time').time() * 1000}) + "\n")
    except:
        pass
    # #endregion
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy=eval_strategy,
        logging_dir=os.path.join(args.output_dir, "logs"),
        save_strategy=save_strategy,
        dataloader_num_workers=0,  # Reduced from 4 to 0 to save memory (prevents parallel data loading)
        logging_steps=args.logging_steps,
        eval_steps=eval_steps if eval_strategy == "steps" else None,
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        load_best_model_at_end=load_best_model,
        metric_for_best_model="eval_loss" if load_best_model else None,
        remove_unused_columns=False,
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        dataloader_pin_memory=False,  # Disable pin_memory to save memory
        optim="adamw_8bit" if args.use_8bit_optimizer else "adamw_torch",  # Use 8-bit optimizer if available
    )
    
    # #region agent log
    try:
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H3", "location": "LoRA_Training.py:after_training_args", "message": "TrainingArguments created", "data": {"use_ddp": use_ddp, "device_map_strategy": device_map_strategy if 'device_map_strategy' in locals() else None, "local_rank_env": os.environ.get("LOCAL_RANK", "not_set")}, "timestamp": __import__('time').time() * 1000}) + "\n")
    except:
        pass
    # #endregion
    
    # #region agent log
    log_memory("LoRA_Training.py:before_trainer_init", "Memory before Trainer initialization", "MEM6")
    # #endregion
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # #region agent log
    log_memory("LoRA_Training.py:before_training", "Memory before training starts", "MEM7")
    # #endregion
    
    # Check available memory before training
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        free = total - reserved
        
        print(f"\n{'='*60}")
        print(f"GPU Memory Analysis")
        print(f"{'='*60}")
        print(f"Total GPU Memory: {total:.2f} GB")
        print(f"Currently Reserved: {reserved:.2f} GB ({reserved/total*100:.1f}%)")
        print(f"Currently Allocated: {allocated:.2f} GB")
        print(f"Free Memory: {free:.2f} GB")
        print(f"\nWhy memory usage is so high:")
        print(f"  1. Base Model (Qwen2.5-VL-7B FP16): ~17-20 GB")
        print(f"     - Model weights stored in FP16")
        print(f"  2. 12 Images per Sample (CRITICAL MEMORY KILLER):")
        print(f"     - Vision encoder processes ALL 12 images simultaneously")
        print(f"     - Each image: ~1024x1024 pixels → large activation tensors")
        print(f"     - Vision encoder activations: ~5-10 GB per batch")
        print(f"     - Image pixel_values tensors: ~2-4 GB")
        print(f"  3. Optimizer States (AdamW):")
        print(f"     - Momentum: ~17-20 GB (if full model)")
        print(f"     - Variance: ~17-20 GB (if full model)")
        print(f"     - With LoRA: Only trainable params (~0.1-0.5 GB)")
        print(f"  4. Gradients: ~17-20 GB during backward pass")
        print(f"  5. Intermediate Activations: ~5-10 GB (stored for backprop)")
        print(f"\nEstimated Breakdown:")
        print(f"  Model weights: ~20 GB")
        print(f"  Vision processing (12 images): ~10-15 GB")
        print(f"  Activations + Gradients: ~25-30 GB")
        print(f"  Total: ~55-65 GB (exceeds 40GB GPU!)")
        print(f"\nKey Issue: Processing 12 images through vision encoder simultaneously")
        print(f"         is extremely memory-intensive, even with batch_size=1")
        
        # Multi-GPU information
        if num_gpus > 1:
            estimated_per_gpu = (total * num_gpus) / num_gpus  # Simplified, but shows distribution
            print(f"\n{'='*60}")
            print(f"Multi-GPU Training Available ({num_gpus} GPUs)")
            print(f"{'='*60}")
            print(f"Model Parallelism: Model layers split across {num_gpus} GPUs")
            print(f"Data Parallelism: Trainer uses DistributedDataParallel")
            print(f"Total GPU Memory: ~{total * num_gpus:.1f} GB across {num_gpus} GPUs")
            print(f"Expected per-GPU usage: ~{total / num_gpus:.1f} GB per GPU (model split)")
            print(f"Memory Benefit: Model weights distributed, reducing per-GPU load")
            print(f"{'='*60}\n")
        else:
            print(f"\n💡 TIP: Use multiple GPUs to distribute memory:")
            print(f"   - Request multiple GPUs in your PBS script: ngpus=2 or ngpus=4")
            print(f"   - The model will automatically split across GPUs")
            print(f"   - Trainer will use data parallelism for faster training")
            print(f"{'='*60}\n")
        
        # Warn if memory is very low
        if free < 2.0:  # Less than 2GB free
            print(f"⚠️  CRITICAL: Very low GPU memory ({free:.2f} GB free).")
            print("\nImmediate actions needed:")
            print("  1. Use MULTIPLE GPUs (recommended):")
            print("     - Request 2-4 GPUs in PBS: ngpus=2 or ngpus=4")
            print("     - Model will auto-distribute across GPUs")
            print("  2. Set: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
            print("  3. Use 8-bit optimizer (install bitsandbytes):")
            print("     pip install bitsandbytes")
            print("     Then add: --use_8bit_optimizer")
            print("  4. Consider using 3B model instead: --model_name Qwen/Qwen2.5-VL-3B-Instruct")
            print("  5. Increase gradient_accumulation_steps to 8 or 16")
            print("  6. Ensure batch_size=1 and dataloader_num_workers=0")
        
        # Clear cache before training
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Train with error handling
    try:
        trainer.train()
        # #region agent log
        log_memory("LoRA_Training.py:after_training", "Memory after training completes", "MEM8")
        # #endregion
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        print(f"Partial checkpoints may be available in: {args.output_dir}")
        raise
    
    # Save final model
    try:
        trainer.model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"\nTraining complete. Model and processor saved to: {args.output_dir}")
    except Exception as e:
        print(f"\nWarning: Failed to save final model: {e}")
        print(f"Checkpoints may still be available in: {args.output_dir}")
        raise
    
if __name__ == "__main__":
    main()

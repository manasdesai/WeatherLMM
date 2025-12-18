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

# loading our dataset from our manifest file.
class ImageTextDataset(Dataset):
    def __init__(self, csv_path:str, max_samples: int = None, image_size: int = None):
        """
        Initialize dataset from CSV.
        
        Args:
            csv_path: Path to CSV manifest file
            max_samples: Maximum number of samples to load (None = all samples). Useful for quick development.
        """
        self.df = pd.read_csv(csv_path)
        self.image_size = image_size
        
        # Limit dataset size if specified (for quick development)
        # Shuffle BEFORE taking samples to ensure temporal diversity
        if max_samples is not None and max_samples > 0:
            original_size = len(self.df)
            # Shuffle to get diverse samples across time periods
            self.df = self.df.sample(n=min(max_samples, original_size), random_state=42).reset_index(drop=True)
            print(f"Limited dataset to {len(self.df)} samples (from {original_size} total, shuffled for temporal diversity)")
        
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
        
        # Filter out invalid samples (missing images) before training
        original_len = len(self.df)
        self.df = self._filter_valid_samples(self.df)
        filtered_count = original_len - len(self.df)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} samples with missing images (from {original_len} total). Remaining: {len(self.df)} valid samples.")
    
    def _filter_valid_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out rows where any image file is missing or invalid.
        Returns a new DataFrame with only valid samples.
        """
        valid_indices = []
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            is_valid = True
            
            try:
                if self.use_semicolon_format:
                    # Check semicolon-separated format
                    image_paths_str = row["image_paths"]
                    if pd.isna(image_paths_str) or not image_paths_str:
                        is_valid = False
                    else:
                        image_paths = [path.strip() for path in str(image_paths_str).split(';') if path.strip()]
                        if len(image_paths) != 12:
                            is_valid = False
                        else:
                            # Check all paths exist
                            for image_path in image_paths:
                                if pd.isna(image_path) or not image_path or not os.path.exists(str(image_path)):
                                    is_valid = False
                                    break
                    
                    # Check text is valid
                    if is_valid:
                        text = row["target_text"]
                        if pd.isna(text) or not str(text).strip():
                            is_valid = False
                else:
                    # Check individual column format
                    for col in self.image_columns:
                        image_path = row[col]
                        if pd.isna(image_path) or not image_path or not os.path.exists(str(image_path)):
                            is_valid = False
                            break
                    
                    # Check text is valid
                    if is_valid:
                        text = row["text"]
                        if pd.isna(text) or not str(text).strip():
                            is_valid = False
            except Exception as e:
                # If any error occurs during validation, skip this sample
                is_valid = False
            
            if is_valid:
                valid_indices.append(idx)
        
        return df.iloc[valid_indices].reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx:int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        our_weather_images = []
        image_paths_list = []

        def _load_image(path: str) -> Image.Image:
            """Load an image and optionally resize to a fixed square size.

            Resizing all 12 images to a smaller resolution (e.g., 448x448 or 512x512)
            drastically reduces GPU memory usage in the vision tower while keeping
            the rest of the training pipeline consistent.
            """
            img = Image.open(str(path)).convert("RGB")
            if self.image_size is not None and self.image_size > 0:
                img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
            return img
        
        if self.use_semicolon_format:
            # Parse semicolon-separated image paths
            image_paths_str = row["image_paths"]
            if pd.isna(image_paths_str) or not image_paths_str:
                raise ValueError(f"image_paths is empty or NaN in row {idx} (should have been filtered)")
            image_paths = [path.strip() for path in str(image_paths_str).split(';') if path.strip()]
            if len(image_paths) != 12:
                raise ValueError(f"Expected 12 image paths, got {len(image_paths)} in row {idx} (should have been filtered)")
            
            for image_path in image_paths:
                # Note: Files should already be validated in __init__, but check again in case files were deleted
                if pd.isna(image_path) or not image_path:
                    raise ValueError(f"Image path is empty in row {idx} (should have been filtered)")
                if not os.path.exists(str(image_path)):
                    raise FileNotFoundError(f"Image file was deleted after dataset initialization: {image_path} (row {idx})")
                try:
                    our_weather_images.append(_load_image(image_path))
                    image_paths_list.append(image_path)
                except Exception as e:
                    raise RuntimeError(f"Failed to load image {image_path} (row {idx}): {e}")
            
            text = row["target_text"]
            if pd.isna(text) or not str(text).strip():
                raise ValueError(f"Text is empty or NaN in row {idx} (should have been filtered)")
        else:
            # Use individual columns
            for col in self.image_columns:
                image_path = row[col]
                # Note: Files should already be validated in __init__, but check again in case files were deleted
                if pd.isna(image_path) or not image_path:
                    raise ValueError(f"Image path is empty in column {col}, row {idx} (should have been filtered)")
                if not os.path.exists(str(image_path)):
                    raise FileNotFoundError(f"Image file was deleted after dataset initialization: {image_path} (column: {col}, row {idx})")
                try:
                    our_weather_images.append(_load_image(image_path))
                    image_paths_list.append(image_path)
                except Exception as e:
                    raise RuntimeError(f"Failed to load image {image_path} (column: {col}, row {idx}): {e}")
            text = row["text"]
            if pd.isna(text) or not str(text).strip():
                raise ValueError(f"Text is empty or NaN in row {idx} (should have been filtered)")

        return {"image": our_weather_images, "image_paths": image_paths_list, "text": text}

# Mapping of variable names to human-readable descriptions
VARIABLE_DESCRIPTIONS = {
    "t_z_1000": "Temperature and geopotential height at 1000 hPa",
    "t_z_850": "Temperature and geopotential height at 850 hPa",
    "t_z_700": "Temperature and geopotential height at 700 hPa",
    "t_z_500": "Temperature and geopotential height at 500 hPa",
    "t_z_200": "Temperature and geopotential height at 200 hPa",
    "t2m_wind10m": "2-meter temperature and 10-meter wind",
    "thickness_mslp": "500-1000 hPa thickness and mean sea level pressure",
    "uv_rh_1000": "Wind and relative humidity at 1000 hPa",
    "uv_rh_850": "Wind and relative humidity at 850 hPa",
    "uv_rh_700": "Wind and relative humidity at 700 hPa",
    "uv_rh_500": "Wind and relative humidity at 500 hPa",
    "uv_rh_200": "Wind and relative humidity at 200 hPa",
}

def extract_variable_name(image_path: str) -> str:
    """
    Extract variable name from image filename.
    
    Examples:
        t_z_1000_0000_12_20200101.1.png -> t_z_1000
        uv_rh_500_1200_12_20200101.1.png -> uv_rh_500
        t2m_wind10m_0000_12_20200101.1.png -> t2m_wind10m
    """
    filename = os.path.basename(image_path)
    # Remove extension and split by underscore
    # Pattern: {variable}_{init_time}_{lead_time}_{date}.1.png
    parts = filename.replace('.1.png', '').replace('.png', '').split('_')
    
    # Variable name is everything before the init_time (which is 4 digits)
    # Find where the 4-digit time starts
    variable_parts = []
    for part in parts:
        if len(part) == 4 and part.isdigit():
            # Found the init_time, stop here
            break
        variable_parts.append(part)
    
    variable_name = '_'.join(variable_parts)
    return variable_name

# Defining our datacollector along with masking.
@dataclass
class WeatherDataCollectorImageText:
    
    processor: AutoProcessor
    
    def _build_content_with_descriptions(self, images: List[Image.Image], image_paths: List[str]) -> List[Dict[str, Any]]:
        """Build content array with interleaved image descriptions and images."""
        content = []
        
        # Add introductory text
        content.append({
            "type": "text",
            "text": (
                "You are a skilled weather forecasting system operating on behalf of the National "
                "Weather Service's Weather Prediction Center. Given a set of numerical weather "
                "prediction model output images for a +12 hour forecast, produce a short-range "
                "synoptic-scale weather forecast for the continental United States over the next 1 "
                "to 2 days. Your forecast will be relied upon by millions across the country, so it "
                "is critical to be careful and accurate. Think deeply about each weather variable "
                "and their relationships, recalling principles of quasi-geostrophic meteorology, "
                "to ensure your forecast is physically consistent before generating your final answer.\n\n"
                "The following 12 forecast maps are provided:\n"
            ),
        })
        
        # Add each image with its description
        for i, (img, img_path) in enumerate(zip(images, image_paths), 1):
            # Extract variable name from path
            variable_name = extract_variable_name(img_path)
            description = VARIABLE_DESCRIPTIONS.get(
                variable_name, 
                f"Forecast map {i}: {variable_name}"
            )
            
            # Add description text
            content.append({
                "type": "text",
                "text": f"Map {i}: {description} ({img_path})",
            })
            
            # Add the image
            content.append({
                "type": "image",
                "image": img,
            })
        
        # Add final instruction
        content.append({
            "type": "text",
            "text": "\nForecast begins below.\n\nValid",
        })
        
        return content
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        weather_images_batch = [item["image"] for item in batch]
        image_paths_batch = [item.get("image_paths", []) for item in batch]
        text = [item["text"] for item in batch]
        
        full_text = []
        user_only_messages = []

        for idx, (images, image_paths, t) in enumerate(zip(weather_images_batch, image_paths_batch, text)):
            # Build content with interleaved descriptions
            content = self._build_content_with_descriptions(images, image_paths)
            
            user_message = {
                "role": "user",
                "content": content,
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

        # This is for our user only prompts where we're getting our prompt length in tokens.
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
        
        # This is the full message with everything included, the user + assistant.
        full_messages = self.processor.apply_chat_template(
            full_text,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # flattening the images for the batch processor.
        # batch size x 12 images for each.
        
        all_weather_images = []
        for images_per_sample in weather_images_batch:
            all_weather_images.extend(images_per_sample)
        
        model_inputs = self.processor(
            text=full_messages,
            images=all_weather_images, # passing all of the images in the batch flattened.
            padding=True,
            # return PyTorch tensors
            return_tensors="pt",
        )
        
        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()
        
        # Masking the users tokens so we the assistant tokens contribute to our loss.
        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = -100  # Masking the prompt part
            
        model_inputs["labels"] = labels
        
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
        "--logging_dir",
        type=str,
        default=None,
        help="Directory for TensorBoard logs. Defaults to {output_dir}/logs if not specified.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help=(
            "Optional square resize for all input images before feeding them to Qwen's "
            "vision encoder (e.g., 448 or 512). "
            "Smaller values significantly reduce GPU memory usage. "
            "If None, original image resolution is used."
        ),
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
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory (can interact poorly with some PEFT setups).",
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
    train_dataset = ImageTextDataset(
        args.train_csv,
        max_samples=args.max_train_samples,
        image_size=args.image_size,
    )
    if len(train_dataset) == 0:
        raise ValueError(f"Training dataset is empty: {args.train_csv}")
    print(f"Training samples: {len(train_dataset)}")
    if args.max_train_samples:
        print(f"  (Limited from full dataset for quick development)")
    
    eval_dataset = None
    if args.eval_csv:
        print("Loading evaluation dataset...")
        eval_dataset = ImageTextDataset(
            args.eval_csv,
            max_samples=args.max_eval_samples,
            image_size=args.image_size,
        )
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
    
    # Check available GPUs (for logging only)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.2f} GB)")
    else:
        num_gpus = 0
        print("No CUDA GPUs available")

    # Load model directly to GPU (don't use device_map="auto" which can offload to CPU)
    # This is critical for training speed - device_map="auto" can offload model to CPU
    print(f"Loading model: {args.model_name}")
    print("Loading model directly to GPU (not using device_map='auto' to avoid CPU offloading)...")
    try:
        # Determine dtype: prefer bfloat16 for training (better than float16)
        if args.bf16:
            model_dtype = torch.bfloat16
        elif args.fp16:
            model_dtype = torch.float16
        else:
            # Default to bfloat16 if available, otherwise float32
            model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=model_dtype,
            # Don't use device_map - load directly to GPU for full speed
        )
        
        # Explicitly move to GPU
        if torch.cuda.is_available():
            model = model.to("cuda:0")
            # Enable CUDA optimizations for faster training
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        
        # Clear cache after model loading
        torch.cuda.empty_cache()
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
        # Target common projection MLP + attention modules in Qwen2.5-VL
        # so that at least some parameters require gradients for LoRA.
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    
    # Apply LoRA with validation
    try:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # Sanity check: ensure we actually have trainable parameters.
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError(
                "LoRA configuration resulted in 0 trainable parameters. "
                "This will cause the loss tensor to have no grad_fn and break backprop. "
                "Check that `target_modules` match the Qwen2.5-VL architecture, or "
                "update `peft`/`transformers` to a compatible version."
            )
        
        # Verify model is fully on GPU after LoRA
        if torch.cuda.is_available():
            all_params = list(model.parameters())
            gpu_params = [p for p in all_params if p.device.type == "cuda"]
            cpu_params = [p for p in all_params if p.device.type == "cpu"]
            
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"\nGPU Memory after LoRA: {allocated:.2f} GB allocated")
            print(f"Parameters on GPU: {len(gpu_params)}/{len(all_params)}")
            
            if cpu_params:
                print(f"WARNING: {len(cpu_params)} parameters are on CPU!")
                print(f"Moving remaining parameters to GPU...")
                model = model.to("cuda:0")
                # Double-check: move all parameters explicitly
                for param in model.parameters():
                    if param.device.type != "cuda":
                        param.data = param.data.to("cuda:0")
                allocated_after = torch.cuda.memory_allocated() / 1024**3
                print(f"After fix: {allocated_after:.2f} GB allocated")
            else:
                print(f"✓ All parameters are on GPU")
        
        # Clear cache after LoRA
        torch.cuda.empty_cache()
    except Exception as e:
        raise RuntimeError(f"Failed to apply LoRA configuration: {e}. Check that target_modules exist in the model.")
    
    data_collator = WeatherDataCollectorImageText(processor=processor)
    
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
        logging_dir=args.logging_dir if args.logging_dir is not None else os.path.join(args.output_dir, "logs"),
        save_strategy=save_strategy,
        dataloader_num_workers=4,  # Use 4 workers for parallel data loading (faster than 0)
        logging_steps=args.logging_steps,
        eval_steps=eval_steps if eval_strategy == "steps" else None,
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        load_best_model_at_end=load_best_model,
        metric_for_best_model="eval_loss" if load_best_model else None,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_pin_memory=True,  # Enable pin_memory for faster CPU→GPU transfers
        optim="adamw_8bit" if args.use_8bit_optimizer else "adamw_torch",  # Use 8-bit optimizer if available
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Check available memory before training
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        free = total - reserved
        
        print(f"\n{'='*60}")
        print(f"GPU Memory Analysis")
        print(f"{'='*60}")
        print(f"Total GPU Memory (GPU 0): {total:.2f} GB")
        print(f"Currently Reserved: {reserved:.2f} GB ({reserved/total*100:.1f}%)")
        print(f"Currently Allocated: {allocated:.2f} GB")
        print(f"Free Memory: {free:.2f} GB")
        
        # Warn if memory is very low
        if free < 2.0:  # Less than 2GB free
            print(f"WARNING: Very low GPU memory ({free:.2f} GB free).")
        
        # Clear cache before training
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Train with error handling
    try:
        trainer.train()
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

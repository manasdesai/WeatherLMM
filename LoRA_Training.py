import os
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
    def __init__(self, csv_path:str):
        self.df = pd.read_csv(csv_path)
        
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
            image_paths = [path.strip() for path in image_paths_str.split(';')]
            
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")
                our_weather_images.append(Image.open(image_path).convert("RGB"))
            
            text = row["target_text"]
        else:
            # Use individual columns
            for col in self.image_columns:
                image_path = row[col]
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")
                our_weather_images.append(Image.open(image_path).convert("RGB"))
            text = row["text"]

        return {"image": our_weather_images, "text": text}
    
# Defining our datacollector along with masking.
@dataclass
class WeatherDataCollectorImageText:
    
    processor: AutoProcessor
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        weather_images_batch = [item["image"] for item in batch]
        text = [item["text"] for item in batch]
        
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
        default=500,
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
    return parser.parse_args()

def main():
    args = parse_args()
        
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=(torch.float16 if args.fp16 else None),
            device_map="auto",
        )

    processor = AutoProcessor.from_pretrained(args.model_name)
    
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
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    train_dataset = ImageTextDataset(args.train_csv)
    eval_dataset = ImageTextDataset(args.eval_csv) if args.eval_csv else None
    data_collator = WeatherDataCollectorImageText(processor=processor)
    
    # #region agent log
    import inspect
    try:
        with open(log_path, 'a') as f:
            sig = inspect.signature(TrainingArguments.__init__)
            params = list(sig.parameters.keys())
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "LoRA_Training.py:317", "message": "TrainingArguments parameters check", "data": {"has_evaluation_strategy": "evaluation_strategy" in params, "has_eval_strategy": "eval_strategy" in params, "all_params": params[:20]}, "timestamp": __import__('time').time() * 1000}) + "\n")
    except Exception as e:
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
        eval_strategy="steps" if eval_dataset is not None else "no",
        logging_dir=os.path.join(args.output_dir, "logs"),
        save_strategy="steps",
        dataloader_num_workers=4,
        logging_steps=args.logging_steps,
        eval_steps=200,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_steps=args.max_steps,
        load_best_model_at_end=True if eval_dataset is not None else False,
        metric_for_best_model="eval_loss",
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    trainer.model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Training complete. Model and processor saved to,", args.output_dir)
    
if __name__ == "__main__":
    main()

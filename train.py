"""
Adapter-based fine-tuning pipeline for Qwen2.5-VL.

Strategy summary:
 - Load pretrained Qwen2.5-VL vision->seq model and processor
 - Freeze all backbone parameters
 - Attach a small adapter MLP that operates on decoder hidden states before
   the LM head (we wrap the model's output_embeddings with a module that
   applies the adapter). This means only adapter parameters are trained.
 - Train using standard cross-entropy loss computed by the model (labels)

Usage (simple):
  python train.py --data-csv ./data/manifest.csv --output_dir ./checkpoints --epochs 3

Dataset manifest format (CSV): each row should contain: nc_path,prompt,target_text
 - nc_path: path to the NetCDF file (or a PNG image path if you pre-exported images)
 - prompt: the prompt text that will be combined with the image
 - target_text: the reference forecast text used as training label

This script is intentionally conservative: it implements a lightweight
adapter strategy that is easy to run on CPU/GPU and safe to start with.

"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModelForVision2Seq, AutoProcessor
try:
    # Optional imports for QLoRA / bitsandbytes pathway
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

from weather_utils import netcdf_to_image
from PIL import Image


class Adapter(nn.Module):
    """Small residual adapter applied to decoder hidden states.

    hidden -> hidden (via bottleneck) so parameter count is small.
    """

    def __init__(self, hidden_size: int, adapter_dim: int = 256):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(adapter_dim, hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (batch, seq_len, hidden_size)
        return self.up(self.act(self.down(hidden)))


class WrappedLMHead(nn.Module):
    """Wrap original lm_head and apply adapter to hidden states before projection."""

    def __init__(self, orig_lm_head: nn.Module, adapter: Adapter):
        super().__init__()
        self.orig = orig_lm_head
        self.adapter = adapter

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # apply adapter in residual form then call original lm head
        adapted = hidden_states + self.adapter(hidden_states)
        return self.orig(adapted)


class WeatherDataset(Dataset):
    """Simple dataset backed by a CSV manifest.

    Each row: nc_path,target_text
    nc_path may also be an image path (png/jpg) if you pre-exported charts.
    """

    def __init__(self, manifest_csv: str, img_size: Tuple[int, int] = (1024, 768)):
        self.records = []
        with open(manifest_csv, newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                self.records.append(r)

        self.img_size = img_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        path = rec.get('nc_path') or rec.get('image')
        target = rec.get('target_text', '')

        # Load image: try to treat path as NetCDF first, fall back to image file
        try:
            image, meta = netcdf_to_image(path)
        except Exception:
            image = Image.open(path).convert('RGB')

        return image, target


def collate_fn(batch: List[Tuple[Image.Image, str]], processor: AutoProcessor, device: torch.device):
    images, targets = zip(*batch)

    # Use a fixed internal prompt (not stored in manifest). The user requested
    # that the manifest not include prompts; the model will still receive a
    # small generation instruction identical for all examples.
    INTERNAL_PROMPT = (
        "You are a meteorologist. Provide a concise, professional forecast based on the provided chart."
    )

    # Build conversation with both user prompt and assistant response for training
    conversations = []
    for img, target in zip(images, targets):
        conversations.append([
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': img},
                    {'type': 'text', 'text': INTERNAL_PROMPT},
                ],
            },
            {
                'role': 'assistant',
                'content': [
                    {'type': 'text', 'text': target},
                ],
            }
        ])

    # Apply chat template to get full text including response
    texts = [processor.apply_chat_template(conv, tokenize=False) for conv in conversations]

    # Process images and text together
    model_inputs = processor(
        text=texts,
        images=list(images),
        padding=True,
        return_tensors='pt'
    )

    # For causal LM training, labels are the same as input_ids
    # The model will internally shift them for next-token prediction
    model_inputs['labels'] = model_inputs['input_ids'].clone()

    # Mask out the prompt tokens so loss is only computed on the assistant response
    # This requires finding where the assistant response starts in each sequence
    for idx, conv in enumerate(conversations):
        # Get the prompt-only text (without assistant response)
        prompt_text = processor.apply_chat_template([conv[0]], tokenize=False, add_generation_prompt=True)
        prompt_tokens = processor(text=prompt_text, images=[images[idx]], return_tensors='pt')
        prompt_length = prompt_tokens['input_ids'].shape[1]

        # Mask prompt tokens in labels (set to -100 so they're ignored in loss)
        model_inputs['labels'][idx, :prompt_length] = -100

    for k, v in list(model_inputs.items()):
        if isinstance(v, torch.Tensor):
            model_inputs[k] = v.to(device)

    return model_inputs


def freeze_backbone(model: AutoModelForVision2Seq):
    for p in model.parameters():
        p.requires_grad = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', '--data-csv', dest='manifest', required=True, help='CSV manifest with nc_path,prompt,target_text')
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct', help='pretrained model id')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--adapter_dim', type=int, default=256)
    # QLoRA / LoRA options
    parser.add_argument('--use_qlora', action='store_true', help='Use QLoRA (bitsandbytes + PEFT) for fine-tuning')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--lora_target_modules', type=str, default='q_proj,k_proj,v_proj,o_proj',
                        help='Comma-separated list of target modules for LoRA')
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f'Loading model {args.model} on {device}...')

    model = None
    processor = None

    # QLoRA / PEFT branch
    if args.use_qlora:
        if args.device == 'cpu':
            raise RuntimeError('QLoRA requires a GPU device. Set --device cuda')

        if BitsAndBytesConfig is None:
            raise ImportError('BitsAndBytes support not available. Install recent transformers and bitsandbytes.')

        # Load model in 4-bit via BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )

        model = AutoModelForVision2Seq.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

        try:
            from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
        except Exception as e:
            raise ImportError('PEFT not installed. Install `peft` and `bitsandbytes` to use --use_qlora') from e

        model = prepare_model_for_kbit_training(model)

        target_modules = [m.strip() for m in args.lora_target_modules.split(',') if m.strip()]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type='SEQ_2_SEQ_LM',
        )

        model = get_peft_model(model, lora_config)

        # When using device_map='auto' the model is already on devices; do not call .to(device)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f'Number of trainable parameters (LoRA): {sum(x.numel() for x in trainable_params)}')
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    else:
        # Adapter-based fine-tuning (default)
        model = AutoModelForVision2Seq.from_pretrained(args.model, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

        # Freeze backbone
        freeze_backbone(model)

        # Build adapter and wrap lm_head
        hidden_size = getattr(model.config, 'd_model', None) or getattr(model.config, 'hidden_size', None)
        # print(f"hidden size is",hidden_size) #2048
        if hidden_size is None:
            # fallback: try to inspect lm_head weight
            try:
                lm = model.get_output_embeddings()
                hidden_size = lm.in_features
            except Exception:
                raise RuntimeError('Unable to infer model hidden size for adapter. Please set manually in code.')

        adapter = Adapter(hidden_size=hidden_size, adapter_dim=args.adapter_dim)

        # Wrap output embeddings
        orig_lm = model.get_output_embeddings()
        wrapped_lm = WrappedLMHead(orig_lm, adapter)
        model.set_output_embeddings(wrapped_lm)

        # Move trainable parameters (adapter + wrapped lm params) to device
        model.to(device)

        # Optimizer: only adapter & wrapped_lm parameters are trainable
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f'Number of trainable parameters: {sum(x.numel() for x in trainable_params)}')

        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # Dataset and dataloader
    dataset = WeatherDataset(args.manifest)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, processor, device))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()

    print(f'\nStarting training with {len(dataset)} examples...')
    print(f'Batch size: {args.batch_size}, Total batches per epoch: {len(dataloader)}\n')

    for epoch in range(args.epochs):
        print(f'{"="*60}')
        print(f'Epoch {epoch+1}/{args.epochs}')
        print(f'{"="*60}')
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            # batch is a dict already moved to device and may contain pixel_values and input_ids and labels
            outputs = model(**batch)

            # Prefer outputs.loss if model computes it; otherwise compute from logits
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            else:
                logits = outputs.logits
                labels = batch.get('labels')
                # shift for decoder if needed
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if (step + 1) % 10 == 0:
                avg_loss = total_loss / (step + 1)
                current_loss = loss.item()
                print(f'  Step {step+1}/{len(dataloader)} | Current loss: {current_loss:.4f} | Avg loss: {avg_loss:.4f}')

        epoch_loss = total_loss / (step + 1)
        print(f'\n{"="*60}')
        print(f'Epoch {epoch+1} completed | Average loss: {epoch_loss:.4f}')
        print(f'{"="*60}\n')

        # Save checkpoint for the chosen fine-tuning method
        if args.use_qlora:
            peft_dir = output_dir / f'peft_epoch{epoch+1}'
            peft_dir.mkdir(parents=True, exist_ok=True)
            # PEFT models provide a `save_pretrained` helper
            try:
                model.save_pretrained(str(peft_dir))
                print(f'  Saved PEFT adapter to: {peft_dir}')
            except Exception:
                # Fallback: save state_dict
                torch.save({'state_dict': model.state_dict()}, peft_dir / 'state_dict.pt')
                print(f'  Saved PEFT state_dict to: {peft_dir / "state_dict.pt"}')
        else:
            adapter_path = output_dir / f'adapter_epoch{epoch+1}.pt'
            torch.save({'adapter_state_dict': adapter.state_dict()}, adapter_path)
            print(f'  Saved adapter checkpoint: {adapter_path}')

    print('Training finished')


if __name__ == '__main__':
    main()

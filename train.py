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

import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModelForVision2Seq, AutoProcessor

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

    Each row: nc_path,prompt,target_text
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
        prompt = rec.get('prompt', '')
        target = rec.get('target_text', '')

        # Load image: try to treat path as NetCDF first, fall back to image file
        try:
            image, meta = netcdf_to_image(path)
        except Exception:
            image = Image.open(path).convert('RGB')

        return image, prompt, target


def collate_fn(batch: List[Tuple[Image.Image, str, str]], processor: AutoProcessor, device: torch.device):
    images, prompts, targets = zip(*batch)

    # Prepare messages that match the processor's expected chat template
    messages = []
    for img, p, t in zip(images, prompts, targets):
        messages.append({
            'role': 'user',
            'content': [
                {'type': 'image', 'image': img},
                {'type': 'text', 'text': p},
            ],
        })

    # The processor used in qwen_inference expects apply_chat_template; we will
    # use it to produce the input text and also pass text_target to get labels.
    texts = [processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in messages]

    # `text_target` is a common name for target texts in HF processors; use if supported.
    try:
        model_inputs = processor(
            text=texts,
            images=list(images),
            text_target=list(targets),
            padding=True,
            return_tensors='pt'
        )
    except TypeError:
        # Fallback: encode prompts as input and encode targets separately
        model_inputs = processor(
            text=texts,
            images=list(images),
            padding=True,
            return_tensors='pt'
        )
        with processor.as_target_processor():
            labels = processor(text=list(targets), padding=True, return_tensors='pt')
            # labels.key: input_ids or input_ids
            model_inputs['labels'] = labels['input_ids']

    # Move tensors to device
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
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f'Loading model {args.model} on {device}...')
    model = AutoModelForVision2Seq.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # Freeze backbone
    freeze_backbone(model)

    # Build adapter and wrap lm_head
    hidden_size = getattr(model.config, 'd_model', None) or getattr(model.config, 'hidden_size', None)
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

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
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
                avg = total_loss / (step + 1)
                print(f'  step {step+1} loss={avg:.4f}')

        epoch_loss = total_loss / (step + 1)
        print(f'Epoch {epoch+1} completed, avg loss={epoch_loss:.4f}')

        # Save adapter checkpoint
        adapter_path = output_dir / f'adapter_epoch{epoch+1}.pt'
        torch.save({'adapter_state_dict': adapter.state_dict()}, adapter_path)
        print(f'  Saved adapter checkpoint: {adapter_path}')

    print('Training finished')


if __name__ == '__main__':
    main()

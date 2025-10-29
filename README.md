# WeatherLMM
Course project for the course CMSC 723: Natural Language Processing

Adapter fine-tuning for Qwen2.5-VL
================================

Overview
--------
This repository includes a minimal adapter-based fine-tuning pipeline in `train.py`.

Key points:
- The pretrained Qwen2.5-VL backbone is frozen.
- A small residual adapter MLP is applied to decoder hidden states before the
  LM head. Only adapter parameters are trained.

Why this approach
------------------
- Low memory and fast to iterate: only a small number of new parameters are trained.
- Safe: backbone remains unchanged so you avoid catastrophic forgetting.

Quick start
-----------
1. Create a CSV manifest with columns: `nc_path,prompt,target_text`.
   - `nc_path` may be a path to a NetCDF `.nc` file (the script will convert to an image)
     or to a PNG image if you exported charts offline.
2. Install dependencies from `requirements.txt` (or create a virtualenv).
3. Run training:

```bash
python train.py --manifest ./data/manifest.csv --output_dir ./checkpoints --epochs 3 --batch_size 2
```

Notes and next steps
--------------------
- The manifest must contain meaningful `target_text` forecasts aligned with the
  corresponding image + prompt. Training will be only as good as those labels.
- If you want to try LoRA or prompt-tuning later, there are commented hints in
  `requirements.txt` (PEFT/bitsandbytes). Those approaches can provide larger
  parameter efficiency but require additional dependency setup.

This README gives a minimal workflow; see `train.py` for implementation details.
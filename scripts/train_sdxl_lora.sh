#!/usr/bin/env bash
# SDXL DreamBooth-LoRA on a 16 GB T4. Override via env vars.
set -euo pipefail
: "${DATA_DIR:=/kaggle/input/my-images}"
: "${TRIGGER:=a photo of sks_dog}"
: "${OUT_DIR:=/kaggle/working/lora}"
: "${STEPS:=1500}"
: "${RANK:=16}"

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
  --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
  --instance_data_dir="$DATA_DIR" --caption_column=text \
  --instance_prompt="$TRIGGER" --resolution=1024 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 --rank="$RANK" --max_train_steps="$STEPS" \
  --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam \
  --checkpointing_steps=250 --validation_prompt="$TRIGGER on the moon" \
  --output_dir="$OUT_DIR" --seed=42

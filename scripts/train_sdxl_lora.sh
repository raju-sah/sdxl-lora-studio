#!/usr/bin/env bash
# SDXL DreamBooth-LoRA (Mithila / Madhubani Style) on a 16 GB T4. Override via env vars.
set -euo pipefail
: "${DATA_DIR:=/kaggle/input/mithila-lora-images}"
: "${TRIGGER:=in sks_mithila style}"
: "${VALIDATION_PROMPT:=in sks_mithila style, two peacocks perched on a flowering tree branch with lotus blossoms and a geometric border}"
: "${OUT_DIR:=/kaggle/working/lora}"
: "${STEPS:=1500}"
: "${RANK:=32}"
: "${LR:=1e-4}"
: "${WARMUP:=100}"
: "${SNR_GAMMA:=5.0}"

accelerate launch --num_processes=1 train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
  --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
  --dataset_name="$DATA_DIR" --caption_column=text \
  --instance_prompt="$TRIGGER" --resolution=1024 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --learning_rate="$LR" --lr_scheduler=cosine --lr_warmup_steps="$WARMUP" \
  --snr_gamma="$SNR_GAMMA" --rank="$RANK" --max_train_steps="$STEPS" \
  --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam \
  --checkpointing_steps=250 --validation_prompt="$VALIDATION_PROMPT" \
  --output_dir="$OUT_DIR" --seed=42

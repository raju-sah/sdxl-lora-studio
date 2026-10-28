# PLAN — LoRA for Image Generation (SDXL)

> Project 3 of 3 · Difficulty ⭐⭐ · GPU: 2–8 h Kaggle T4 · Week 2 (alongside RAG evals)

## What it is

Image generation uses **diffusion models**, not LLMs. We take an existing image model (SDXL)
and train a **LoRA** — small low-rank matrices (~10–200 MB) injected into the attention layers
— that teaches it a new **subject** (face, pet, product) or **style** (Thangka, Mithila painting,
own sketch style) from 10–50 images.

## Base model choice (16 GB T4)

| Base model | Size | LoRA on T4? | Notes |
|---|---|---|---|
| SD 1.5 | ~1B | ✅ Easy, ~20–40 min | Older quality, great for learning |
| **SDXL 1.0** | ~3.5B | ✅ **Recommended**, ~1–2 h | Good quality @1024 px; needs fp16 tricks |
| SD 3.5 Medium | ~2.5B | ⚠️ Possible | Less community tooling |
| FLUX.1-dev / schnell | 12B | ⚠️ Hard | Normally 24 GB; 4-bit + heavy tricks, slow. Dev = non-commercial license |

## Requirements

- **Data:** 15–40 high-quality images (varied angles, lighting, backgrounds), center-cropped
  to 1024×1024.
- **Captions:** one `.txt` per image with a trigger word, e.g. `a photo of sks_dog sitting on grass`.
  Auto-caption with BLIP / Florence-2, then hand-edit.
- **Tools:** 🤗 diffusers `train_dreambooth_lora_sdxl.py` (simplest) — alt: kohya-ss sd-scripts, SimpleTuner.
- **T4 memory tricks:** `--mixed_precision=fp16`, `--gradient_checkpointing`, `--use_8bit_adam`,
  batch size 1, and the `madebyollin/sdxl-vae-fp16-fix` VAE (stock SDXL VAE → NaNs/black images in fp16).

## Open decisions (for /grill-me when we get here)

- [ ] **Subject or style?** and which one (pet / self / product / Thangka / Mithila / sketches)
- [ ] Image source & rights (own photos vs public-domain art)
- [ ] Captioner: BLIP vs Florence-2
- [ ] Demo host: HF Space (ZeroGPU) vs Civitai

## Steps

- [ ] 1. **Collect & clean** images → `data/raw/`. Run `src/lora_studio/prepare.py` (crop/resize
      to 1024², drop blurry/duplicates) → `data/processed/`. Upload as private Kaggle dataset:
      `kaggle datasets create -p data/processed` (metadata in `data/dataset-metadata.json`).
- [ ] 2. **Caption** (`caption.py`): auto-caption, insert trigger word, hand-edit.
- [ ] 3. **Train** on Kaggle (`kaggle/train/`, `scripts/train_sdxl_lora.sh`):
  ```bash
  accelerate launch train_dreambooth_lora_sdxl.py \
    --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
    --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
    --instance_data_dir=/kaggle/input/my-images --caption_column=text \
    --instance_prompt="a photo of sks_dog" --resolution=1024 \
    --train_batch_size=1 --gradient_accumulation_steps=4 \
    --learning_rate=1e-4 --rank=16 --max_train_steps=1500 \
    --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam \
    --checkpointing_steps=250 --validation_prompt="a photo of sks_dog on the moon" \
    --output_dir=/kaggle/working/lora --seed=42
  ```
- [ ] 4. **Evaluate** (`grid.py`, `metrics.py`): fixed grid (same prompts in `prompts/eval_prompts.txt`
      + same seeds) at each checkpoint 250/500/…/1500 **and** base model without LoRA.
      Optional metrics: CLIP score (prompt adherence), DINO similarity to training images
      (subject fidelity). Watch for **overfitting** (every image looks like a training photo,
      prompt ignored) → pick the best checkpoint, not the last.
- [ ] 5. **Host**: LoRA → HF Hub with model card + sample images; Gradio Space demo
      (`space/app.py`, ZeroGPU or CPU-offload) and/or Civitai.

## Definition of done

- [ ] Checkpoint × prompt comparison grid (incl. no-LoRA baseline) in README
- [ ] CLIP / DINO scores per checkpoint (table or plot) and chosen-checkpoint rationale
- [ ] LoRA on HF Hub with model card; live Gradio Space link

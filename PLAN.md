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

- **Concept:** **Mithila (Madhubani) traditional folk painting** Style LoRA (`in sks_mithila style`).
- **Data:** 25–30 high-quality public-domain / CC-licensed images from Wikimedia Commons & open museum APIs (varied *bharni* colored fills and *kachni* line-hatching), deduplicated (pHash), filtered for sharpness (Laplacian variance), center-cropped to 1024×1024, and logged in `data/sources.csv`.
- **Captions:** one `.txt` per image + `metadata.jsonl` sidecar with trigger prefix `in sks_mithila style, <content>`. Auto-captioned with `microsoft/Florence-2-base` (`<DETAILED_CAPTION>`) with generic medium/style words stripped so the style is absorbed into the trigger.
- **Tools:** 🤗 diffusers `train_dreambooth_lora_sdxl.py`.
- **T4 memory & convergence tricks:** `--mixed_precision=fp16`, `--gradient_checkpointing`, `--use_8bit_adam`, batch size 1 (`gradient_accumulation_steps=4`), UNet-only (`rank=32`, frozen text encoders), `--snr_gamma=5.0`, `--lr_scheduler=cosine --lr_warmup_steps=100`, and `madebyollin/sdxl-vae-fp16-fix` VAE (stock SDXL VAE → NaNs/black images in fp16).

## Open decisions (resolved via /grill-me)

- [x] **Subject or style?** Style LoRA — **Mithila / Madhubani traditional folk painting** (`sks_mithila`)
- [x] **Image source & rights:** 25–30 public-domain / CC-licensed images from Wikimedia Commons & open museum collections with license manifest (`data/sources.csv`) + pHash dedupe & blur filtering in `prepare.py`
- [x] **Captioner:** `microsoft/Florence-2-base` (`<DETAILED_CAPTION>`) with style-word stripping, prepending `in sks_mithila style, `, exporting `.txt` + `metadata.jsonl`
- [x] **Training hyperparameters (16 GB T4):** 1024×1024, `rank=32`, UNet-only (frozen text encoders), 1,500 steps (`checkpointing_steps=250`), LR `1e-4` cosine w/ 100 warmup steps, `snr_gamma=5.0`
- [x] **Evaluation & checkpoint selection:** 6 prompts (2 traditional, 2 regional, 2 OOD) × 7 checkpoints (`base, 250, 500, 750, 1000, 1250, 1500`) + CLIP content adherence score & DINOv2 style similarity (2 seeds for metrics, seed 0 for visual grid)
- [x] **Demo host:** Hugging Face Hub model card (`sdxl-mithila-lora`) + Gradio Space (`space/app.py`) with `@spaces.GPU` (ZeroGPU) + `enable_model_cpu_offload()` fallback and interactive LoRA scale slider (`0.0–1.2`)

## Steps

- [x] 1. **Collect & clean** images → `data/raw/` + `data/sources.csv`. Run `src/lora_studio/prepare.py` (crop/resize
      to 1024², drop blurry/duplicates) → `data/processed/`. Upload as Kaggle dataset (`buggeddev/mithila-lora-images`):
      `kaggle datasets create -p data/processed` (metadata in `data/dataset-metadata.json`).
- [x] 2. **Caption** (`caption.py`): auto-caption with `Florence-2-base`, strip style words, prepend `in sks_mithila style, `, write `.txt` + `metadata.jsonl`, and inspect/hand-edit.
- [ ] 3. **Train** on Kaggle (`kaggle/train/`, `scripts/train_sdxl_lora.sh`):
  ```bash
  accelerate launch train_dreambooth_lora_sdxl.py \
    --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
    --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
    --dataset_name=/kaggle/input/mithila-lora-images --caption_column=text \
    --instance_prompt="in sks_mithila style" --resolution=1024 \
    --train_batch_size=1 --gradient_accumulation_steps=4 \
    --learning_rate=1e-4 --lr_scheduler=cosine --lr_warmup_steps=100 \
    --snr_gamma=5.0 --rank=32 --max_train_steps=1500 \
    --mixed_precision=fp16 --gradient_checkpointing --use_8bit_adam \
    --checkpointing_steps=250 \
    --validation_prompt="in sks_mithila style, two peacocks perched on a flowering tree branch with lotus blossoms and a geometric border" \
    --output_dir=/kaggle/working/lora --seed=42
  ```
- [ ] 4. **Evaluate** (`grid.py`, `metrics.py`): fixed grid (6 prompts in `prompts/eval_prompts.txt` across 2 traditional, 2 regional, and 2 OOD scenes) at each checkpoint `250/500/750/1000/1250/1500` **and** base SDXL (`0`) without LoRA.
      Compute **CLIP content score** (prompt adherence with style prefix stripped) and **DINOv2 similarity** to training images (style fidelity) across 2 seeds `[0, 1]` (seed `0` for visual grid) to catch overfitting and pick the Pareto-optimal checkpoint.
- [ ] 5. **Host**: Upload best LoRA checkpoint to HF Hub (`sdxl-mithila-lora`) with model card + sample images; build Gradio Space demo (`space/app.py`, `@spaces.GPU` ZeroGPU with CPU-offload fallback and LoRA scale slider `0.0–1.2`).

## Definition of done

- [ ] Checkpoint × prompt comparison grid (incl. no-LoRA baseline) in README
- [ ] CLIP / DINO scores per checkpoint (table or plot) and chosen-checkpoint rationale
- [ ] LoRA on HF Hub with model card; live Gradio Space link

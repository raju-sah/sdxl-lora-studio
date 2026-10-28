# NOTES — SDXL LoRA

Running log of decisions, findings, and gotchas. Newest first.

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-10-28 | **Hosting & Demo:** HF Hub model card (`rajucode/sdxl-mithila-lora`) + Gradio Space (`space/app.py`) with `@spaces.GPU` (ZeroGPU) and automatic `enable_model_cpu_offload()` fallback + LoRA scale slider (`0.0–1.2`); Kaggle dataset slug set to `rajucode/mithila-lora-images` | ZeroGPU provides fast (~6s) free inference on HF Spaces while CPU-offload fallback guarantees the exact same app runs on Kaggle/Colab T4 or CPU environments without crashing; scale slider lets users interactively compare base SDXL (`0.0`) vs Mithila LoRA (`0.85–1.0`). |
| 2026-10-28 | **Evaluation & Checkpoint Selection:** 6 structured prompts (2 traditional motifs, 2 regional scenes, 2 hard OOD prompts) × 7 checkpoints (`base, 250, 500, 750, 1000, 1250, 1500`) + CLIP content adherence (`clip-vit-large-patch14`, trigger stripped) + DINOv2 style similarity (`dinov2-base`) across 2 seeds `[0, 1]` (seed `0` for visual grid) | OOD prompts (`astronaut on the moon`, `cat beside teacup`) expose composition memorization when late checkpoints ignore the prompt and draw traditional fish/peacocks. Pairing CLIP content adherence (stripped of style prefix) with DINOv2 training-set similarity detects the exact Pareto sweet spot (~12–15 min eval runtime on T4). |
| 2026-10-28 | **T4 Hyperparameters:** 1024×1024, `rank=32`, UNet-only (frozen text encoders), 1,500 steps (`checkpointing_steps=250`), `batch_size=1`, `grad_accum=4`, LR `1e-4` cosine (`warmup=100`), `snr_gamma=5.0`, fp16 + `madebyollin/sdxl-vae-fp16-fix` + 8-bit Adam + grad checkpointing | Mithila art combines dense *kachni* line-hatching, *bharni* pigment fills, and geometric borders; `rank=32` captures richer spatial texture than `rank=16` at negligible VRAM cost (<100 MB extra). Freezing both SDXL text encoders saves ~3 GB VRAM (~12 GB peak on 16 GB T4) and prevents text-encoder overfitting so novel OOD prompts still compose cleanly. Min-SNR (`5.0`) + cosine LR stabilize convergence. |
| 2026-10-28 | **Trigger & Captioning:** Trigger prefix `in sks_mithila style` + `microsoft/Florence-2-base` (`<DETAILED_CAPTION>`) with style-word stripping + `.txt` and `metadata.jsonl` export | In a Style LoRA, any uncaptioned content gets absorbed into the trigger word. `Florence-2-base` describes subjects, animals, flora, and layout much more thoroughly than BLIP-1, while stripping generic medium words (`painting`, `illustration`, `folk art`, `Madhubani`) forces the visual style onto `sks_mithila`. Exporting `metadata.jsonl` ensures `diffusers` (`--dataset_name` / `imagefolder`) actually reads per-image `--caption_column=text`. |
| 2026-10-28 | **Data Sourcing & Quality Control:** Curate 25–30 public-domain / CC-licensed images via script from Wikimedia Commons & open museum APIs, log provenance in `data/sources.csv`, and add perceptual hash deduplication + Laplacian blur filtering to `prepare.py` | 25–30 diverse paintings avoid memorizing individual compositions (<15 images) while fitting a ~1.5h T4 budget. Strict Wikimedia/museum provenance (`CC0`, `PD`, `CC-BY`, `CC-BY-SA`) logged in `data/sources.csv` keeps the dataset and public HF weights copyright-clean. |
| 2026-10-28 | **Concept Choice:** Style LoRA on **Mithila / Madhubani traditional folk painting** | High visual distinctiveness (double-line *kachni* hatching, flat *bharni* fills, geometric borders) where base SDXL is weak, yielding a dramatic before/after checkpoint grid; zero PII/privacy concerns for a public portfolio repo and Gradio Space. |
| 2026-10-28 | Repo scaffolded; SDXL 1.0 + diffusers DreamBooth-LoRA as default | Best quality that fits a 16 GB T4 comfortably |

## GPU budget tracker

| Date | Run | Steps | Rank | LR | GPU h | Best ckpt | Notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## Reference notes

- **Trigger word:** use a rare token (`sks_dog`, `ohwx`) so it doesn't collide with existing concepts.
- **Style LoRA vs subject LoRA:** style → caption the *content* of each image (not the style) so
  the style gets absorbed by the trigger; subject → caption background/pose, keep subject = trigger.
- **fp16 VAE:** always `madebyollin/sdxl-vae-fp16-fix` on T4.
- **diffusers script** lives at `examples/dreambooth/train_dreambooth_lora_sdxl.py`; pin the
  diffusers version matching the script (install from the same git tag).
- **Overfitting signals:** backgrounds copied from training set, prompt words ignored, faces
  identical across seeds. Lower steps or LR, or raise data diversity.

## Gotchas

- _(fill in as you hit them)_

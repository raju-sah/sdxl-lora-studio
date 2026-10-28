# NOTES — SDXL LoRA

Running log of decisions, findings, and gotchas. Newest first.

## Decision log

| Date | Decision | Why |
|---|---|---|
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

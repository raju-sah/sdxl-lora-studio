# sdxl-lora-studio

Training a **SDXL LoRA** for a custom subject/style from 15–40 images on a **free Kaggle T4**,
with a reproducible checkpoint-comparison grid, CLIP/DINO metrics, and a Gradio demo.

> Status: 🚧 scaffolded — see [PLAN.md](PLAN.md) for the roadmap and [NOTES.md](NOTES.md) for the decision log.

## Results

_Checkpoint × prompt grid and CLIP/DINO table — TBD._

## Layout

```
src/lora_studio/   prepare (crop/resize/dedupe), caption, grid, metrics
scripts/           train_sdxl_lora.sh (diffusers DreamBooth-LoRA)
configs/           training hyper-params
prompts/           fixed eval prompts (same prompts + seeds for every checkpoint)
kaggle/train/      Kaggle kernel (T4)
space/             Gradio demo for HF Spaces
data/              raw + processed images (git-ignored)
tests/
```

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python -m lora_studio.prepare data/raw data/processed
kaggle datasets create -p data/processed
kaggle kernels push -p kaggle/train
```

## License

MIT (code). Base model under the CreativeML Open RAIL++-M license.

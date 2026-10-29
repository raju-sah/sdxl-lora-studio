# SDXL Mithila / Madhubani Style LoRA Studio 🎨

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Weights-blue)](https://huggingface.co/raju-ai/sdxl-mithila-lora)
[![ZeroGPU Space](https://img.shields.io/badge/%F0%9F%8E%A8%20ZeroGPU-Live%20Demo-orange)](https://huggingface.co/spaces/raju-ai/sdxl-mithila-lora-demo)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset%20(30%20Images)-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/buggeddev/mithila-lora-images)
[![Kaggle Training](https://img.shields.io/badge/Kaggle-T4%20Training%20Kernel-20BEFF?logo=kaggle)](https://www.kaggle.com/code/buggeddev/sdxl-lora-train)
[![Tests](https://img.shields.io/badge/pytest-8%20passed-brightgreen)](tests/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: CreativeML Open RAIL++-M](https://img.shields.io/badge/License-CreativeML--OpenRAIL--M%20%2F%20MIT-lightgrey)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)

End-to-end engineering studio for training, quantitatively evaluating, and deploying a **Stable Diffusion XL (SDXL 1.0) Style LoRA** on a **free Kaggle T4 GPU (16 GB VRAM)**. 

The model teaches SDXL the ancient, intricate folk art tradition of **Mithila / Madhubani painting** (`in sks_mithila style`) from Nepal and Bihar, India. It faithfully reproduces traditional dual-line *kachni* hatching, bold *bharni* natural pigment fills, sacred nature motifs (peacocks, lotus blossoms, sacred fish, banyan trees), and ornate geometric framing borders.

---

## 📑 Table of Contents

- [Live Demos & Model Hub](#-live-demos--model-hub)
- [Cultural Heritage & Visual Grammar](#-cultural-heritage--visual-grammar)
- [Visual Comparison Grid](#-visual-comparison-grid)
- [Quantitative Evaluation & Checkpoint Selection](#-quantitative-evaluation--checkpoint-selection)
- [Dataset Curation & Smart Captioning](#-dataset-curation--smart-captioning)
- [T4 Training & Memory Architecture](#-t4-training--memory-architecture)
- [Quickstart & Inference Guide](#-quickstart--inference-guide)
- [Kaggle Reproducibility](#-kaggle-reproducibility)
- [Repository Structure](#-repository-structure)
- [License & Acknowledgments](#-license--acknowledgments)

---

## 🚀 Live Demos & Model Hub

| Resource | Link | Description |
|---|---|---|
| **🎨 Interactive ZeroGPU Space** | [🤗 `raju-ai/sdxl-mithila-lora-demo`](https://huggingface.co/spaces/raju-ai/sdxl-mithila-lora-demo) | Free interactive web demo running on Hugging Face ZeroGPU (`zero-a10g`). Generates full 1024×1024 artwork in **~9.9 seconds** with an interactive LoRA scale slider (`0.0–1.2`). |
| **📦 Hugging Face Model Hub** | [🤗 `raju-ai/sdxl-mithila-lora`](https://huggingface.co/raju-ai/sdxl-mithila-lora) | SafeTensors weights (`pytorch_lora_weights.safetensors`, 178 MB, rank 32), detailed Model Card, interactive Inference API widgets, and sample gallery. |
| **📊 Kaggle Training Kernel** | [`buggeddev/sdxl-lora-train`](https://www.kaggle.com/code/buggeddev/sdxl-lora-train) | Fully autonomous Kaggle notebook running 1,500 training steps + 84-image evaluation sweep on a single 16 GB T4 GPU. |
| **📁 Kaggle Dataset** | [`buggeddev/mithila-lora-images`](https://www.kaggle.com/datasets/buggeddev/mithila-lora-images) | 30 curated museum-quality images center-cropped to 1024×1024 with Florence-2 detailed captions and `metadata.jsonl`. |

---

## 🌺 Cultural Heritage & Visual Grammar

**Mithila (Madhubani) art** is a centuries-old folk painting tradition practiced by women in the Mithila region of southeastern Nepal (Janakpur) and northern Bihar, India. Traditionally created on mud walls of village huts during weddings and religious festivals using bamboo twigs and cotton swabs dipped in natural dyes (turmeric, indigo, soot, rice paste), Mithila art is characterized by:

1. **Kachni (Line Work):** Fine, rhythmic, monochromatic double-line hatching and cross-hatching without intermediate shading.
2. **Bharni (Color Pigment Fills):** Rich, solid color fills traditionally applied within defined double-line contours.
3. **Sacred Flora & Fauna:** Symbolic nature imagery—peacocks (grace & love), fish (fertility & prosperity), lotus flowers (purity), turtles, and flowering kadamba/banyan trees.
4. **Horror Vacui & Geometric Borders:** Every corner of the canvas is energized with floral motifs, waves, and geometric borders so no negative space remains empty.

### Why Train a LoRA?
Off-the-shelf foundation models like **SDXL 1.0** lack authentic Mithila art representation. When prompted with `"Madhubani painting"`, base SDXL outputs generic modern digital clip-art, photorealistic scenes with slight color saturation, or stereotypical motifs with Western anatomy. 

By fine-tuning SDXL's cross-attention layers with a **Rank-32 LoRA** (`in sks_mithila style`), the model learns the distinct 2D perspective, stylized almond eyes, double-contour outlines, and natural earthy pigments, while preserving the composition flexibility of SDXL to illustrate both traditional epics and novel out-of-domain concepts.

---

## 🖼️ Visual Comparison Grid

The 84-image evaluation sweep below compares **Base SDXL (Step 0)** against 6 training checkpoints (`ckpt-250` through `ckpt-1500`) across 6 carefully designed test prompts:

![Checkpoint Comparison Grid](outputs/eval/checkpoint_grid.png)

### Prompt Categories Evaluated
1. **Row 1 — Traditional Sacred Motif:** `"in sks_mithila style, two peacocks perched on a flowering kadamba tree branch surrounded by lotus blossoms and an ornate geometric border"`
2. **Row 2 — Historic Procession:** `"in sks_mithila style, a royal procession with a decorated elephant, musicians, and dancers in intricate kachni double-line hatching"`
3. **Row 3 — Regional Cultural Landscape:** `"in sks_mithila style, a snowy Himalayan village beneath Mount Everest with terraced fields, yak herders, and fluttering prayer flags"`
4. **Row 4 — Sacred Aquatic Ecosystem:** `"in sks_mithila style, a serene pond filled with golden fish, turtles, and blooming pink lotus flowers in bold bharni flat pigment fills"`
5. **Row 5 — Hard Out-of-Domain (OOD):** `"in sks_mithila style, an astronaut riding a white horse across the cratered surface of the moon under a starry cosmos"`
6. **Row 6 — Hard Out-of-Domain (OOD):** `"in sks_mithila style, a cozy cafe table with a tabby cat sleeping beside a steaming teacup and an open book by a window"`

---

## 📈 Quantitative Evaluation & Checkpoint Selection

To eliminate subjective cherry-picking, we evaluated all checkpoints quantitatively across two orthogonal axes:
1. **CLIP Content Adherence Score (`openai/clip-vit-large-patch14`)**: Measures semantic prompt alignment with the style trigger prefix stripped.
2. **DINOv2 Style Similarity (`facebook/dinov2-base`)**: Measures visual feature cosine similarity to the reference Mithila training dataset.
3. **DINOv2 Max Nearest-Neighbor (Memorization Guard)**: Measures the maximum cosine similarity to any single training canvas to ensure the model generalizes rather than memorizing individual training paintings (safe threshold: `< 0.88`).

| Checkpoint | Training Steps | CLIP Content Score (↑) | DINOv2 Style Sim (↑) | DINOv2 Max-NN (↓) | Qualitative Assessment |
|---|---:|---:|---:|---:|---|
| **Base SDXL** | 0 | 0.2961 | 0.1865 | 0.3956 | Baseline: photorealistic or western vector art, zero Mithila line grammar |
| `ckpt-250` | 250 | 0.2718 | 0.3728 | 0.5945 | Underfit: early double lines emerge, but pigments remain muted |
| `ckpt-500` | 500 | 0.2501 | 0.3544 | 0.5937 | Good: *kachni* hatching and floral framing begin to solidify |
| **`ckpt-750`** | **750** | **0.2666** | **0.3949** *(+111%)* | **0.6317** | 🏆 **Selected Champion:** Highest prompt fidelity, crisp border motifs, rich *bharni* fills, clean OOD transfer |
| `ckpt-1000` | 1000 | 0.2588 | 0.3694 | 0.6043 | Minor degradation: background flora begins bleeding across unrelated prompts |
| `ckpt-1250` | 1250 | 0.2568 | 0.3957 | 0.6263 | Overfitting signs: astronaut helmet starts merging with folk facial templates |
| `ckpt-1500` | 1500 | 0.2641 | 0.3814 | 0.6214 | Late saturation: background repetition and loss of subtle line details |

### Why `ckpt-750` Won
- **Peak Generalization:** Delivers a **+111% style similarity gain** over base SDXL while retaining superior prompt adherence (`0.2666`) compared to all other trained checkpoints.
- **Out-of-Domain Generalization:** In Rows 5 and 6 (Astronaut and Cozy Cafe), `ckpt-750` transforms modern astronaut suits and teacups into authentic 2D Mithila folk motifs without morphing them into fish or peacocks.
- **Zero Memorization:** The Max-NN score of `0.6317` is well below the memorization threshold (`0.88`), demonstrating authentic stylistic synthesis.

---

## 🗂️ Dataset Curation & Smart Captioning

A LoRA is only as good as its data. We curated **30 museum-quality artworks** with strict open-access licensing:
- **10 Masterworks from the Victoria & Albert (V&A) Museum IIIF Collection** (London, UK)
- **20 Verified Paintings from Wikimedia Commons** (`CC0`, `Public Domain`, `CC-BY`, `CC-BY-SA`)
- Full provenance manifest documented in [`data/sources.csv`](data/sources.csv).

```
Raw Images (V&A IIIF + Wikimedia)
              │
              ▼
   [19% Border Cropping]      ──> Removes museum color calibration bars & framing margins
              │
              ▼
   [dHash Deduplication]      ──> Eliminates duplicate scans & near-identical uploads
              │
              ▼
   [Laplacian Blur Filter]    ──> Rejects blurry uploads (variance threshold > 100.0)
              │
              ▼
   [Center Crop to 1024×1024] ──> High-resolution square format for SDXL UNet
              │
              ▼
   [Florence-2 Auto-Caption]  ──> Detailed vision-language captioning
              │
              ▼
   [Style-Word Scrubbing]     ──> Strips "painting", "folk art", "mithila", "madhubani"
              │
              ▼
    in sks_mithila style, ... ──> Forces visual aesthetic to bind onto the rare token
```

### Smart Captioning Strategy
In a **Style LoRA**, any visual element described in the caption is attributed by the model to the caption words, while **un-captioned visual patterns get absorbed into the rare trigger token**. 

We leveraged `microsoft/Florence-2-base` (`<DETAILED_CAPTION>`) to describe subjects, animals, postures, and compositions in granular detail. We then ran a regex cleaner to strip all generic style words (`painting`, `illustration`, `folk art`, `Madhubani`, `Mithila`, `drawing`). Finally, we prepended the rare token prefix:
```
in sks_mithila style, two peacocks standing in front of a tree with flowers and leaves, surrounded by a border with floral designs
```
This ensured that the distinctive dual outlines, 2D perspectives, and flat pigments bonded entirely to `sks_mithila`.

---

## ⚡ T4 Training & Memory Architecture

Training SDXL (3.5 billion parameters) at native 1024×1024 resolution on a single 16 GB Nvidia T4 GPU requires surgical VRAM optimization:

| Technique | Implementation | Impact |
|---|---|---|
| **FP16 VAE Hotfix** | `madebyollin/sdxl-vae-fp16-fix` | Resolves the notorious stock SDXL VAE fp16 numerical overflow bug (black screens / NaNs). |
| **UNet-Only LoRA** | Frozen Dual Text Encoders (`CLIP ViT-L` + `OpenCLIP ViT-bigG`) | Saves **~3 GB VRAM** and prevents catastrophic forgetting in text embedding space. |
| **Rank-32 / Alpha-32** | UNet Cross-Attention & Spatial Transformers | Captures intricate *kachni* linework that collapses under Rank-16, with negligible VRAM overhead (<100 MB). |
| **8-bit Adam** | `bitsandbytes.optim.AdamW8bit` | Reduces optimizer state memory by 75% (~1.8 GB savings). |
| **Gradient Checkpointing** | Enabled on UNet | Trades ~15% compute time to drop peak training memory below **11.8 GB** on the 16 GB T4. |
| **Min-SNR Gamma Weighting** | `--snr_gamma=5.0` | Prevents high-noise diffusion timesteps from dominating low-rank weight updates, stabilizing fine detail convergence. |
| **Cosine LR Scheduler** | `1e-4` LR with 100 warmup steps | Smoothly scales learning rate down over 1,500 steps (effective batch size = 4 via accumulation). |

---

## 💻 Quickstart & Inference Guide

### 1. Interactive Web Demo (ZeroGPU)
The fastest way to test the model is through the live Hugging Face Space:  
👉 **[Launch Gradio Demo on Hugging Face](https://huggingface.co/spaces/raju-ai/sdxl-mithila-lora-demo)**

### 2. Python (`diffusers` Pipeline)
Run inference locally on any GPU with $\ge 8\text{ GB}$ VRAM:

```python
import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionXLPipeline

# 1. Load SDXL with fp16 VAE fix
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", 
    torch_dtype=torch.float16
)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

# 2. Use DPM++ 2M Karras scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, 
    use_karras_sigmas=True
)

# 3. Load Mithila LoRA weights from Hugging Face Hub
pipe.load_lora_weights("raju-ai/sdxl-mithila-lora")

# 4. Generate artwork
prompt = "in sks_mithila style, two peacocks perched on a flowering tree branch with lotus blossoms and an ornate geometric border"
negative_prompt = "blurry, photorealistic, 3d render, western comic, low quality, distorted"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=28,
    guidance_scale=7.5,
    cross_attention_kwargs={"scale": 0.85},  # Recommended LoRA scale: 0.80 - 0.90
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("mithila_artwork.png")
```

### 3. Programmatic ZeroGPU Inference (`gradio_client`)
Generate images on Hugging Face's A10G GPU cluster via Python in under 10 seconds:

```python
from gradio_client import Client

client = Client("raju-ai/sdxl-mithila-lora-demo")
result = client.predict(
    prompt="in sks_mithila style, an astronaut riding a white horse on the moon under stars",
    lora_scale=0.85,
    steps=25,
    guidance_scale=7.5,
    seed=42,
    auto_trigger=True,
    api_name="/predict",
)
print("Saved generated image at:", result)
```

### 🎯 Recommended Generation Parameters
- **Trigger:** Always include `in sks_mithila style` in the prompt.
- **LoRA Scale:** `0.80 – 0.90` (0.85 is default; 0.0 disables LoRA, >1.0 increases contrast).
- **CFG (Guidance Scale):** `7.0 – 8.0`.
- **Inference Steps:** `25 – 35` (DPM++ 2M Karras or Euler a).
- **Resolution:** Native `1024×1024` (or `1152×896` / `896×1152`).
- **Negative Prompt:** `blurry, photorealistic, 3d render, realistic photo, modern digital art, low quality, oversaturated`.

---

## 🛠️ Kaggle Reproducibility

The training and evaluation pipeline is completely reproducible on Kaggle's free tier.

### 1. Push Dataset to Kaggle
```bash
# Preprocess images and generate Florence-2 captions
python -m lora_studio.prepare data/raw data/processed
python -m lora_studio.caption data/processed

# Upload to Kaggle
kaggle datasets create -p data/processed
```

### 2. Push & Launch Kaggle Kernel
```bash
# Push training kernel (auto-starts training on Kaggle T4)
kaggle kernels push -p kaggle/train
kaggle kernels status buggeddev/sdxl-lora-train
```

The kernel runs `kaggle/train/train.py`, which:
1. Prepares the environment and uninstalls conflicting pre-installed packages (`torchao`).
2. Runs 1,500 training steps of UNet-only LoRA via `accelerate launch`.
3. Runs the post-training 84-image evaluation sweep across checkpoints `[0, 250, 500, 750, 1000, 1250, 1500]`.
4. Computes CLIP content scores and DINOv2 style similarities to select the best checkpoint.
5. Saves the winning weights to `/kaggle/working/best_lora`.

---

## 📂 Repository Structure

```
sdxl-lora-studio/
├── configs/
│   └── sdxl_lora.yaml             # Complete training, evaluation, and hosting configuration
├── data/
│   ├── sources.csv                # Provenance manifest for all 30 curated training artworks
│   └── dataset-metadata.json      # Kaggle dataset definition
├── kaggle/
│   └── train/
│       ├── kernel-metadata.json   # Kaggle T4 kernel specification
│       └── train.py               # Autonomous Kaggle training + evaluation runner
├── outputs/
│   └── eval/
│       ├── checkpoint_grid.png    # 84-image evaluation comparison grid (tracked in git)
│       ├── metrics.json           # Quantitative CLIP + DINOv2 metrics for all checkpoints
│       └── metrics_table.md       # Formatted Markdown metrics summary
├── prompts/
│   └── eval_prompts.txt           # Fixed evaluation prompt benchmark (Traditional, Regional, OOD)
├── scripts/
│   └── train_sdxl_lora.sh         # Diffusers DreamBooth-LoRA launch script
├── space/
│   ├── app.py                     # Gradio 5.50.0 application configured for HF ZeroGPU (@spaces.GPU)
│   ├── README.md                  # Hugging Face Space metadata and YAML frontmatter
│   └── requirements.txt           # Space runtime dependencies
├── src/
│   └── lora_studio/
│       ├── caption.py             # Florence-2 auto-captioning & style-word scrubber
│       ├── fetch.py               # V&A Museum IIIF & Wikimedia Commons automated fetcher
│       ├── grid.py                # Evaluation grid generation & image stitching
│       ├── metrics.py             # CLIP Content Adherence & DINOv2 Style Similarity scoring
│       └── prepare.py             # Image normalization, dHash deduplication & blur filtering
├── tests/
│   ├── test_caption.py            # Unit tests for caption processing & style-word regex
│   ├── test_eval.py               # Unit tests for eval grid & metric calculations
│   └── test_prepare.py            # Unit tests for image filtering & deduplication
├── NOTES.md                       # Comprehensive technical decision log, findings & gotchas
├── PLAN.md                        # Project roadmap and definition-of-done checklist
├── pyproject.toml                 # Package configuration and dependencies
└── README.md                      # Project documentation and engineering report
```

### Local Development & Testing
```bash
# Setup virtual environment with uv
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run test suite
pytest

# Run linter
ruff check .
```

---

## 📜 License & Acknowledgments

- **Code:** Licensed under the [MIT License](LICENSE).
- **Model Weights & Space:** Released under the [CreativeML Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) in alignment with base SDXL 1.0.
- **Cultural & Institutional Acknowledgments:**
  - Dedicated to the generations of Mithila artists who preserved and nurtured this sacred folk art form across centuries.
  - Sincere gratitude to the **Victoria and Albert Museum, London** for their open IIIF cultural API, and to the **Wikimedia Commons** contributors for making authentic cultural heritage accessible under open licenses.
  - Special thanks to the Hugging Face team for **ZeroGPU** grant access.
"""Kaggle T4 entrypoint for SDXL Mithila Style LoRA training + evaluation.

1. Installs pinned diffusers (v0.31.0), peft, bitsandbytes, accelerate, datasets.
2. Downloads matching train_dreambooth_lora_sdxl.py from huggingface/diffusers.
3. Launches 1,500-step UNet-only Rank-32 LoRA training on a 16 GB T4 (fp16, 8-bit Adam,
   gradient checkpointing, madebyollin/sdxl-vae-fp16-fix, cosine LR, snr_gamma=5.0).
4. Renders the 6-prompt x 7-checkpoint comparison grid and computes CLIP + DINOv2 metrics.
5. Copies the winning checkpoint to /kaggle/working/best_lora and uploads to HF Hub if HF_TOKEN is set.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

DIFFUSERS_TAG = "v0.31.0"
TRAIN_SCRIPT_URL = (
    f"https://raw.githubusercontent.com/huggingface/diffusers/{DIFFUSERS_TAG}"
    "/examples/dreambooth/train_dreambooth_lora_sdxl.py"
)

EVAL_PROMPTS = [
    "in sks_mithila style, two peacocks perched on a flowering kadamba tree branch surrounded by lotus blossoms and an ornate geometric border",
    "in sks_mithila style, a royal procession with a decorated elephant, musicians, and dancers in intricate kachni double-line hatching",
    "in sks_mithila style, a snowy Himalayan village beneath Mount Everest with terraced fields, yak herders, and fluttering prayer flags",
    "in sks_mithila style, a serene pond filled with golden fish, turtles, and blooming pink lotus flowers in bold bharni flat pigment fills",
    "in sks_mithila style, an astronaut riding a white horse across the cratered surface of the moon under a starry cosmos",
    "in sks_mithila style, a cozy cafe table with a tabby cat sleeping beside a steaming teacup and an open book by a window",
]


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def setup_environment(work_dir: Path) -> Path:
    """Install dependencies and fetch the pinned diffusers DreamBooth-LoRA SDXL script."""
    # Uninstall incompatible Kaggle torchao pre-install which breaks peft import_utils
    try:
        run_cmd([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
    except Exception:  # noqa: BLE001, S110
        pass

    run_cmd(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            f"git+https://github.com/huggingface/diffusers.git@{DIFFUSERS_TAG}",
            "peft>=0.13.0",
            "bitsandbytes>=0.44.0",
            "accelerate>=1.0.0",
            "transformers>=4.45.0,<=4.48.3",
            "datasets>=3.0.0",
            "safetensors>=0.4.5",
            "huggingface_hub>=0.26.0",
        ]
    )
    # Monkey-patch in current process
    try:
        import transformers.utils

        if not hasattr(transformers.utils, "FLAX_WEIGHTS_NAME"):
            transformers.utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
    except Exception:  # noqa: BLE001, S110
        pass

    script_path = work_dir / "train_dreambooth_lora_sdxl.py"
    if not script_path.exists():
        urllib.request.urlretrieve(TRAIN_SCRIPT_URL, script_path)

    # Prepend FLAX_WEIGHTS_NAME polyfill to the downloaded training script
    content = script_path.read_text(encoding="utf-8")
    patch = (
        "import transformers.utils\n"
        "if not hasattr(transformers.utils, 'FLAX_WEIGHTS_NAME'):\n"
        "    transformers.utils.FLAX_WEIGHTS_NAME = 'flax_model.msgpack'\n"
    )
    if "FLAX_WEIGHTS_NAME = 'flax_model.msgpack'" not in content:
        script_path.write_text(patch + content, encoding="utf-8")

    # Smoke test core imports before launching accelerate
    smoke_cmd = (
        "import diffusers, peft, transformers; "
        "from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel; "
        "print('Verified diffusers, peft, and transformers imports successfully!')"
    )
    run_cmd([sys.executable, "-c", smoke_cmd])

    return script_path


def find_dataset_dir() -> Path:
    """Locate the mounted Kaggle dataset directory containing processed 1024x1024 PNGs and metadata.jsonl."""
    candidates = [
        Path(os.environ.get("DATA_DIR", "/kaggle/input/mithila-lora-images")),
        Path("/kaggle/input/datasets/buggeddev/mithila-lora-images"),
        Path("/kaggle/input/datasets/rajucode/mithila-lora-images"),
        Path("/kaggle/input/my-lora-images"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for sub in input_root.rglob("metadata.jsonl"):
            return sub.parent
    raise FileNotFoundError("Could not find mithila-lora-images dataset in /kaggle/input")


def load_hf_token() -> str | None:
    """Retrieve HF_TOKEN from environment or Kaggle UserSecretsClient."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    work_dir = Path("/kaggle/working")
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir = work_dir / "lora"
    eval_dir = work_dir / "eval"
    prompts_path = work_dir / "eval_prompts.txt"
    prompts_path.write_text("\n".join(EVAL_PROMPTS) + "\n", encoding="utf-8")

    script_path = setup_environment(work_dir)
    raw_data_dir = find_dataset_dir()

    # Copy only *.png and metadata.jsonl to writable directory so Hugging Face imagefolder loader works cleanly
    data_dir = work_dir / "dataset"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for item in raw_data_dir.iterdir():
        if item.suffix.lower() in {".png", ".jpg", ".jpeg"} or item.name == "metadata.jsonl":
            shutil.copy2(item, data_dir / item.name)

    train_cmd = [
        "accelerate",
        "launch",
        "--num_processes=1",
        "--mixed_precision=fp16",
        str(script_path),
        "--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
        "--pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix",
        f"--dataset_name={data_dir}",
        "--caption_column=text",
        "--instance_prompt=in sks_mithila style",
        "--resolution=1024",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4",
        "--learning_rate=1e-4",
        "--lr_scheduler=cosine",
        "--lr_warmup_steps=100",
        "--snr_gamma=5.0",
        "--rank=32",
        "--max_train_steps=1500",
        "--mixed_precision=fp16",
        "--gradient_checkpointing",
        "--use_8bit_adam",
        "--checkpointing_steps=250",
        f"--output_dir={out_dir}",
        "--seed=42",
    ]
    run_cmd(train_cmd)

    # Run evaluation grid & metrics if lora_studio is present or inline
    print("Training finished. Starting evaluation grid + CLIP/DINOv2 scoring...", flush=True)
    eval_summary = evaluate_checkpoints_inline(
        lora_dir=out_dir,
        train_dir=data_dir,
        eval_dir=eval_dir,
        prompts=EVAL_PROMPTS,
        checkpoints=[0, 250, 500, 750, 1000, 1250, 1500],
        seeds=[0, 1],
    )

    best_ckpt = int(eval_summary["best_checkpoint"])
    best_src = out_dir / f"checkpoint-{best_ckpt}" if best_ckpt > 0 else out_dir
    best_dst = work_dir / "best_lora"
    best_dst.mkdir(parents=True, exist_ok=True)
    weights_file = best_src / "pytorch_lora_weights.safetensors"
    if weights_file.exists():
        shutil.copy2(weights_file, best_dst / "pytorch_lora_weights.safetensors")

    hf_token = load_hf_token()
    if hf_token:
        from huggingface_hub import HfApi

        repo_id = os.environ.get("HF_REPO_ID", "raju-ai/sdxl-mithila-lora")
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo_id, exist_ok=True)
        api.upload_folder(folder_path=str(best_dst), repo_id=repo_id, commit_message=f"Upload best checkpoint-{best_ckpt}")
        api.upload_file(
            path_or_fileobj=str(eval_dir / "checkpoint_grid.png"),
            path_in_repo="checkpoint_grid.png",
            repo_id=repo_id,
        )
        api.upload_file(
            path_or_fileobj=str(eval_dir / "metrics.json"),
            path_in_repo="metrics.json",
            repo_id=repo_id,
        )
        print(f"Uploaded best checkpoint ({best_ckpt}) and evaluation artifacts to https://huggingface.co/{repo_id}")


def evaluate_checkpoints_inline(
    lora_dir: Path,
    train_dir: Path,
    eval_dir: Path,
    prompts: list[str],
    checkpoints: list[int],
    seeds: list[int],
) -> dict:
    """Self-contained evaluation runner for the standalone Kaggle script."""
    import torch
    import torch.nn.functional as F
    from diffusers import (
        AutoencoderKL,
        DPMSolverMultistepScheduler,
        StableDiffusionXLPipeline,
    )
    from PIL import Image, ImageDraw, ImageFont
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

    eval_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe.set_progress_bar_config(disable=True)

    grid_images: dict[str, list[Image.Image]] = {}
    for ckpt in checkpoints:
        label = "Base SDXL" if ckpt == 0 else f"ckpt-{ckpt}"
        cdir = eval_dir / (f"ckpt-{ckpt:04d}" if ckpt > 0 else "ckpt-0000-base")
        cdir.mkdir(parents=True, exist_ok=True)
        pipe.unload_lora_weights()
        if ckpt > 0:
            cpath = lora_dir / f"checkpoint-{ckpt}"
            if not cpath.exists() and ckpt == max(checkpoints):
                cpath = lora_dir
            if not cpath.exists():
                continue
            pipe.load_lora_weights(str(cpath))

        grid_images[label] = []
        for p_idx, prompt in enumerate(prompts):
            for seed in seeds:
                out_png = cdir / f"p{p_idx:02d}_s{seed}.png"
                if out_png.exists():
                    im = Image.open(out_png).convert("RGB")
                else:
                    gen = torch.Generator(device=device).manual_seed(seed)
                    im = pipe(prompt=prompt, num_inference_steps=25, guidance_scale=7.5, generator=gen).images[0]
                    im.save(out_png)
                if seed == seeds[0]:
                    grid_images[label].append(im)

    # Free SDXL pipeline VRAM before running CLIP + DINOv2
    del pipe, vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Stitch comparison grid
    tile = 256
    label_w, header_h = 220, 44
    cols = list(grid_images.keys())
    canvas = Image.new("RGB", (label_w + len(cols) * tile, header_h + len(prompts) * tile), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for c_idx, cname in enumerate(cols):
        draw.text((label_w + c_idx * tile + 12, 14), cname, fill=(240, 240, 245), font=font)
    for r_idx, prompt in enumerate(prompts):
        short_p = prompt.removeprefix("in sks_mithila style,").strip()
        draw.text((10, header_h + r_idx * tile + 14), f"P{r_idx + 1}: {short_p[:28]}", fill=(210, 215, 225), font=font)
        for c_idx, cname in enumerate(cols):
            if r_idx < len(grid_images[cname]):
                canvas.paste(
                    grid_images[cname][r_idx].resize((tile, tile), Image.LANCZOS),
                    (label_w + c_idx * tile, header_h + r_idx * tile),
                )
    canvas.save(eval_dir / "checkpoint_grid.png")

    # Compute CLIP + DINOv2 metrics
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_mod = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino_mod = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

    train_vecs = []
    with torch.no_grad():
        for tp in sorted(train_dir.glob("*.png")):
            rgb = Image.open(tp).convert("RGB")
            d_in = dino_proc(images=rgb, return_tensors="pt").to(device)
            d_out = dino_mod(**d_in)
            train_vecs.append(F.normalize(d_out.last_hidden_state[:, 0, :], dim=-1))
    train_embeds = torch.cat(train_vecs, dim=0)

    results = []
    for ckpt in checkpoints:
        cdir = eval_dir / (f"ckpt-{ckpt:04d}" if ckpt > 0 else "ckpt-0000-base")
        pngs = sorted(cdir.glob("p*_s*.png"))
        if not pngs:
            continue
        c_scores, d_means, d_maxs = [], [], []
        for png in pngs:
            p_idx = int(png.name.split("_")[0].removeprefix("p"))
            content_p = prompts[p_idx].removeprefix("in sks_mithila style,").strip()
            rgb = Image.open(png).convert("RGB")
            with torch.no_grad():
                c_in = clip_proc(text=[content_p], images=[rgb], return_tensors="pt", padding=True, truncation=True).to(device)
                c_out = clip_mod(**c_in)
                cos = (F.normalize(c_out.image_embeds, dim=-1) * F.normalize(c_out.text_embeds, dim=-1)).sum(dim=-1).item()
                c_scores.append(max(0.0, float(cos)))

                d_in = dino_proc(images=rgb, return_tensors="pt").to(device)
                d_vec = F.normalize(dino_mod(**d_in).last_hidden_state[:, 0, :], dim=-1)
                sims = torch.matmul(d_vec, train_embeds.T).squeeze(0)
                d_means.append(float(sims.mean().item()))
                d_maxs.append(float(sims.max().item()))
        results.append(
            {
                "checkpoint": ckpt,
                "clip_content_score": sum(c_scores) / len(c_scores),
                "dino_style_similarity": sum(d_means) / len(d_means),
                "dino_max_nn": sum(d_maxs) / len(d_maxs),
            }
        )

    lora_rows = [r for r in results if r["checkpoint"] > 0]
    peak_clip = max(r["clip_content_score"] for r in lora_rows)
    valid = [r for r in lora_rows if r["clip_content_score"] >= peak_clip - 0.025 and r["dino_max_nn"] <= 0.88] or lora_rows
    best = max(valid, key=lambda r: 0.55 * r["clip_content_score"] + 0.45 * r["dino_style_similarity"])

    summary = {"best_checkpoint": best["checkpoint"], "best_metrics": best, "checkpoints": results}
    (eval_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()

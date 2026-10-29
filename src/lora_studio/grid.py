"""Generate a fixed prompt x seed grid for the base SDXL model and each LoRA checkpoint,
and stitch a labeled Checkpoint x Prompt comparison sheet.

Usage:
  python -m lora_studio.grid --lora-dir /kaggle/working/lora --prompts prompts/eval_prompts.txt --out-dir outputs/eval
  python -m lora_studio.grid --restitch outputs/eval/checkpoint_grid.png --prompts prompts/eval_prompts.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from lora_studio.config import load_config


def load_prompts(prompts_file: Path) -> list[str]:
    """Read non-empty, non-comment prompt lines from ``prompts_file``."""
    return [
        line.strip()
        for line in prompts_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def wrap_prompt_label(prompt: str, row_idx: int, max_chars: int = 26, max_lines: int = 8) -> str:
    """Strip the trigger prefix and word-wrap a prompt into compact lines for grid row headers."""
    short_prompt = prompt.removeprefix("in sks_mithila style,").removeprefix("in sks_mithila style").strip()
    words = short_prompt.split()
    lines: list[str] = []
    current = f"P{row_idx + 1}:"
    for w in words:
        if len(current) + len(w) + 1 <= max_chars:
            current = f"{current} {w}".strip()
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    return "\n".join(lines[:max_lines])


def stitch_checkpoint_grid(
    grid_images: dict[str, list[Image.Image]],
    prompts: list[str],
    tile_size: int = 256,
    label_width: int = 220,
    header_height: int = 44,
) -> Image.Image:
    """Stitch a labeled table where columns are checkpoints (``Base SDXL``, ``ckpt-250``, ...) and rows are prompts."""
    ckpt_names = list(grid_images.keys())
    n_cols = len(ckpt_names)
    n_rows = len(prompts)

    canvas_w = label_width + n_cols * tile_size
    canvas_h = header_height + n_rows * tile_size
    canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    # Draw column headers
    for col_idx, ckpt_label in enumerate(ckpt_names):
        x0 = label_width + col_idx * tile_size
        draw.text((x0 + 12, 14), ckpt_label, fill=(240, 240, 245), font=font)

    # Draw rows (wrapped prompt label + tiles)
    for row_idx, prompt in enumerate(prompts):
        y0 = header_height + row_idx * tile_size
        wrapped = wrap_prompt_label(prompt, row_idx=row_idx)
        draw.multiline_text((10, y0 + 14), wrapped, fill=(210, 215, 225), font=font, spacing=4)

        for col_idx, ckpt_label in enumerate(ckpt_names):
            imgs = grid_images[ckpt_label]
            if row_idx >= len(imgs):
                continue
            tile = imgs[row_idx].convert("RGB").resize((tile_size, tile_size), Image.LANCZOS)
            x0 = label_width + col_idx * tile_size
            canvas.paste(tile, (x0, y0))

    return canvas


def restitch_existing_grid(
    grid_path: Path,
    prompts: list[str],
    checkpoints: list[int] | None = None,
    tile_size: int = 256,
    label_width: int = 220,
    header_height: int = 44,
) -> Image.Image:
    """Extract lossless tiles from an existing ``checkpoint_grid.png`` and re-render with wrapped prompt labels."""
    if checkpoints is None:
        checkpoints = [0, 250, 500, 750, 1000, 1250, 1500]
    with Image.open(grid_path) as existing:
        rgb = existing.convert("RGB")
        grid_images: dict[str, list[Image.Image]] = {}
        for col_idx, ckpt in enumerate(checkpoints):
            label = "Base SDXL" if ckpt == 0 else f"ckpt-{ckpt}"
            x0 = label_width + col_idx * tile_size
            if x0 + tile_size > rgb.width:
                break
            col_tiles: list[Image.Image] = []
            for row_idx in range(len(prompts)):
                y0 = header_height + row_idx * tile_size
                if y0 + tile_size > rgb.height:
                    break
                col_tiles.append(rgb.crop((x0, y0, x0 + tile_size, y0 + tile_size)))
            grid_images[label] = col_tiles
    return stitch_checkpoint_grid(
        grid_images,
        prompts,
        tile_size=tile_size,
        label_width=label_width,
        header_height=header_height,
    )


def make_grid(
    lora_dir: Path,
    prompts_file: Path,
    out_dir: Path,
    checkpoints: list[int] | None = None,
    seeds: list[int] | None = None,
    grid_seed: int = 0,
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    vae_model: str = "madebyollin/sdxl-vae-fp16-fix",
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    lora_scale: float = 0.85,
    tile_size: int = 256,
) -> Path:
    """Generate evaluation images for base SDXL + all checkpoints and save ``checkpoint_grid.png``."""
    import torch
    from diffusers import (
        AutoencoderKL,
        DPMSolverMultistepScheduler,
        StableDiffusionXLPipeline,
    )

    if checkpoints is None:
        checkpoints = [0, 250, 500, 750, 1000, 1250, 1500]
    if seeds is None:
        seeds = [0, 1]

    prompts = load_prompts(prompts_file)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    vae = AutoencoderKL.from_pretrained(vae_model, torch_dtype=dtype)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        vae=vae,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe.set_progress_bar_config(disable=True)

    grid_images: dict[str, list[Image.Image]] = {}

    for ckpt in checkpoints:
        ckpt_label = "Base SDXL" if ckpt == 0 else f"ckpt-{ckpt}"
        ckpt_subdir = out_dir / (f"ckpt-{ckpt:04d}" if ckpt > 0 else "ckpt-0000-base")
        ckpt_subdir.mkdir(parents=True, exist_ok=True)

        pipe.unload_lora_weights()
        if ckpt > 0:
            ckpt_path = lora_dir / f"checkpoint-{ckpt}"
            if not ckpt_path.exists() and (lora_dir / "pytorch_lora_weights.safetensors").exists() and ckpt == max(checkpoints):
                ckpt_path = lora_dir
            if not ckpt_path.exists():
                print(f"skip missing checkpoint: {ckpt_path}")
                continue
            pipe.load_lora_weights(str(ckpt_path))

        grid_images[ckpt_label] = []
        attn_kwargs = {"scale": float(lora_scale)} if ckpt > 0 else None
        for p_idx, prompt in enumerate(prompts):
            for seed in seeds:
                img_path = ckpt_subdir / f"p{p_idx:02d}_s{seed}.png"
                if img_path.exists():
                    img = Image.open(img_path).convert("RGB")
                else:
                    gen = torch.Generator(device=device).manual_seed(seed)
                    img = pipe(
                        prompt=prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=gen,
                        cross_attention_kwargs=attn_kwargs,
                    ).images[0]
                    img.save(img_path)
                if seed == grid_seed:
                    grid_images[ckpt_label].append(img)
        print(f"completed generation for {ckpt_label}")

    grid_canvas = stitch_checkpoint_grid(grid_images, prompts, tile_size=tile_size)
    grid_out = out_dir / "checkpoint_grid.png"
    grid_canvas.save(grid_out)
    print(f"saved comparison grid to {grid_out}")
    return grid_out


if __name__ == "__main__":
    cfg = load_config()
    eval_cfg = cfg.get("eval", {})
    host_cfg = cfg.get("hosting", {})
    ap = argparse.ArgumentParser(description="Render Checkpoint x Prompt comparison grid.")
    ap.add_argument("--lora-dir", type=Path, help="Directory containing checkpoint-N folders")
    ap.add_argument("--restitch", type=Path, help="Re-stitch an existing checkpoint_grid.png with wrapped labels")
    ap.add_argument("--prompts", type=Path, default=Path("prompts/eval_prompts.txt"), help="Eval prompts file")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/eval"), help="Output directory for renders and grid")
    ap.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=list(eval_cfg.get("checkpoints", [0, 250, 500, 750, 1000, 1250, 1500])),
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=list(eval_cfg.get("seeds", [0, 1])))
    ap.add_argument("--grid-seed", type=int, default=int(eval_cfg.get("grid_seed", 0)))
    ap.add_argument("--steps", type=int, default=int(eval_cfg.get("num_inference_steps", 25)))
    ap.add_argument("--guidance-scale", type=float, default=float(eval_cfg.get("guidance_scale", 7.5)))
    ap.add_argument("--lora-scale", type=float, default=float(host_cfg.get("default_lora_scale", 0.85)))
    a = ap.parse_args()
    if a.restitch:
        canvas = restitch_existing_grid(
            a.restitch,
            prompts=load_prompts(a.prompts),
            checkpoints=a.checkpoints,
        )
        canvas.save(a.restitch)
        print(f"re-stitched {a.restitch}")
    else:
        if a.lora_dir is None:
            ap.error("--lora-dir is required unless --restitch is specified")
        make_grid(
            lora_dir=a.lora_dir,
            prompts_file=a.prompts,
            out_dir=a.out_dir,
            checkpoints=a.checkpoints,
            seeds=a.seeds,
            grid_seed=a.grid_seed,
            base_model=str(cfg.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0")),
            vae_model=str(cfg.get("vae", "madebyollin/sdxl-vae-fp16-fix")),
            num_inference_steps=a.steps,
            guidance_scale=a.guidance_scale,
            lora_scale=a.lora_scale,
        )

"""Compute CLIP content adherence score (trigger stripped) and DINOv2 style similarity
to the Mithila training dataset across checkpoints, and select the best checkpoint before overfitting.

Usage:
  python -m lora_studio.metrics --eval-dir outputs/eval --train-dir data/processed --prompts prompts/eval_prompts.txt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image

from lora_studio.grid import load_prompts


def strip_trigger_prefix(prompt: str, trigger_prefix: str = "in sks_mithila style") -> str:
    """Remove the style trigger prefix from ``prompt`` so CLIP scores pure content adherence."""
    text = prompt.strip()
    prefix = trigger_prefix.strip().rstrip(",")
    if text.lower().startswith(prefix.lower()):
        text = text[len(prefix) :].lstrip(" ,.:;-")
    return text or prompt.strip()


def select_best_checkpoint(results: list[dict], clip_drop_tolerance: float = 0.025) -> dict:
    """Pick the LoRA checkpoint with highest balanced CLIP content + DINOv2 style score before overfitting.

    Overfitting is flagged when a checkpoint's CLIP content score drops by more than
    ``clip_drop_tolerance`` relative to the peak LoRA CLIP score, or when ``dino_max_nn`` > 0.88.
    """
    lora_rows = [r for r in results if int(r.get("checkpoint", 0)) > 0]
    if not lora_rows:
        return results[0] if results else {}

    peak_clip = max(r["clip_content_score"] for r in lora_rows)
    candidates = [
        r
        for r in lora_rows
        if r["clip_content_score"] >= (peak_clip - clip_drop_tolerance) and r.get("dino_max_nn", 0.0) <= 0.88
    ]
    if not candidates:
        candidates = lora_rows

    # Combined score: weight content adherence (0.55) + style fidelity (0.45)
    return max(
        candidates,
        key=lambda r: 0.55 * float(r["clip_content_score"]) + 0.45 * float(r["dino_style_similarity"]),
    )


def format_metrics_markdown(results: list[dict], best_ckpt: int) -> str:
    """Render a GitHub-flavored Markdown table summarizing CLIP & DINOv2 scores per checkpoint."""
    lines = [
        "| Checkpoint | CLIP Content Score ↑ | DINOv2 Style Sim ↑ | DINOv2 Max-NN (Memorization) | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for r in results:
        ckpt = int(r["checkpoint"])
        label = "Base SDXL (0)" if ckpt == 0 else f"`ckpt-{ckpt}`"
        status = "Baseline" if ckpt == 0 else ("⭐ **Selected Best**" if ckpt == best_ckpt else r.get("status", "OK"))
        lines.append(
            f"| {label} | {r['clip_content_score']:.4f} | {r['dino_style_similarity']:.4f} | "
            f"{r.get('dino_max_nn', 0.0):.4f} | {status} |"
        )
    return "\n".join(lines) + "\n"


def score_checkpoint(
    ckpt_dir: Path,
    prompts: list[str],
    train_dino_embeds,
    clip_model,
    clip_processor,
    dino_model,
    dino_processor,
    device: str = "cpu",
    trigger_prefix: str = "in sks_mithila style",
) -> dict[str, float]:
    """Compute mean CLIP content adherence and DINOv2 style similarity for a single checkpoint folder."""
    import torch
    import torch.nn.functional as F

    pngs = sorted(ckpt_dir.glob("p*_s*.png"))
    if not pngs:
        raise FileNotFoundError(f"No rendered p*_s*.png files found in {ckpt_dir}")

    clip_scores: list[float] = []
    dino_mean_sims: list[float] = []
    dino_max_sims: list[float] = []

    for img_path in pngs:
        m = re.match(r"p(\d+)_s(\d+)\.png$", img_path.name)
        p_idx = int(m.group(1)) if m else 0
        prompt = prompts[min(p_idx, len(prompts) - 1)]
        content_prompt = strip_trigger_prefix(prompt, trigger_prefix=trigger_prefix)

        with Image.open(img_path) as im:
            rgb = im.convert("RGB")

        with torch.no_grad():
            clip_inputs = clip_processor(
                text=[content_prompt],
                images=[rgb],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            clip_out = clip_model(**clip_inputs)
            img_emb = F.normalize(clip_out.image_embeds, dim=-1)
            txt_emb = F.normalize(clip_out.text_embeds, dim=-1)
            clip_cos = float((img_emb * txt_emb).sum(dim=-1).item())
            clip_scores.append(max(0.0, clip_cos))

            dino_inputs = dino_processor(images=rgb, return_tensors="pt").to(device)
            dino_out = dino_model(**dino_inputs)
            dino_vec = F.normalize(dino_out.last_hidden_state[:, 0, :], dim=-1)
            sims = torch.matmul(dino_vec, train_dino_embeds.T).squeeze(0)
            dino_mean_sims.append(float(sims.mean().item()))
            dino_max_sims.append(float(sims.max().item()))

    return {
        "clip_content_score": sum(clip_scores) / len(clip_scores),
        "dino_style_similarity": sum(dino_mean_sims) / len(dino_mean_sims),
        "dino_max_nn": sum(dino_max_sims) / len(dino_max_sims),
        "num_images": len(pngs),
    }


def score_all_checkpoints(
    eval_dir: Path,
    train_dir: Path,
    prompts_file: Path,
    clip_model_id: str = "openai/clip-vit-large-patch14",
    dino_model_id: str = "facebook/dinov2-base",
    trigger_prefix: str = "in sks_mithila style",
) -> dict:
    """Evaluate all ``ckpt-*`` subdirectories in ``eval_dir`` and write ``metrics.json`` & ``metrics_table.md``."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = load_prompts(prompts_file)

    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
    clip_model = CLIPModel.from_pretrained(clip_model_id).to(device).eval()

    dino_processor = AutoImageProcessor.from_pretrained(dino_model_id)
    dino_model = AutoModel.from_pretrained(dino_model_id).to(device).eval()

    # Pre-compute DINOv2 embeddings for all training images in train_dir
    train_paths = sorted(p for p in train_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not train_paths:
        raise FileNotFoundError(f"No training images found in {train_dir}")

    train_vecs = []
    with torch.no_grad():
        for tp in train_paths:
            with Image.open(tp) as im:
                rgb = im.convert("RGB")
            d_in = dino_processor(images=rgb, return_tensors="pt").to(device)
            d_out = dino_model(**d_in)
            train_vecs.append(F.normalize(d_out.last_hidden_state[:, 0, :], dim=-1))
    train_dino_embeds = torch.cat(train_vecs, dim=0)

    ckpt_dirs = sorted(p for p in eval_dir.iterdir() if p.is_dir() and p.name.startswith("ckpt-"))
    results: list[dict] = []
    for cdir in ckpt_dirs:
        m = re.search(r"ckpt-(\d+)", cdir.name)
        ckpt_num = int(m.group(1)) if m else 0
        scores = score_checkpoint(
            ckpt_dir=cdir,
            prompts=prompts,
            train_dino_embeds=train_dino_embeds,
            clip_model=clip_model,
            clip_processor=clip_processor,
            dino_model=dino_model,
            dino_processor=dino_processor,
            device=device,
            trigger_prefix=trigger_prefix,
        )
        row = {"checkpoint": ckpt_num, **scores}
        results.append(row)
        print(
            f"ckpt-{ckpt_num:04d}: CLIP={scores['clip_content_score']:.4f} "
            f"DINO={scores['dino_style_similarity']:.4f} (max_nn={scores['dino_max_nn']:.4f})"
        )

    best = select_best_checkpoint(results)
    best_ckpt = int(best.get("checkpoint", 0))

    # Annotate overfitting status on late checkpoints
    peak_clip = max((r["clip_content_score"] for r in results if r["checkpoint"] > 0), default=0.0)
    for r in results:
        if r["checkpoint"] > best_ckpt and r["clip_content_score"] < peak_clip - 0.015:
            r["status"] = "⚠️ Overfitting (CLIP drop)"
        elif r["checkpoint"] < best_ckpt and r["checkpoint"] > 0:
            r["status"] = "Underfit / warming up"
        else:
            r["status"] = "Good"

    md_table = format_metrics_markdown(results, best_ckpt=best_ckpt)
    summary = {
        "best_checkpoint": best_ckpt,
        "best_metrics": best,
        "checkpoints": results,
    }
    (eval_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (eval_dir / "metrics_table.md").write_text(md_table, encoding="utf-8")
    print(f"\nSelected best checkpoint: ckpt-{best_ckpt}")
    print(md_table)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score checkpoints with CLIP and DINOv2.")
    ap.add_argument("--eval-dir", type=Path, default=Path("outputs/eval"), help="Directory of ckpt-* folders")
    ap.add_argument("--train-dir", type=Path, default=Path("data/processed"), help="Processed training images directory")
    ap.add_argument("--prompts", type=Path, default=Path("prompts/eval_prompts.txt"), help="Eval prompts file")
    a = ap.parse_args()
    score_all_checkpoints(eval_dir=a.eval_dir, train_dir=a.train_dir, prompts_file=a.prompts)

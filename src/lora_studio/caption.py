"""Auto-caption images with Florence-2 (or BLIP), strip style leakage words, prepend trigger,
and write both per-image .txt files and a Hugging Face imagefolder metadata.jsonl sidecar.

Usage:
  python -m lora_studio.caption data/processed --trigger "in sks_mithila style"
  python -m lora_studio.caption data/processed --sync-only
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from lora_studio.config import load_config

EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Leading boilerplate phrases produced by Florence-2 / BLIP captioners.
_LEADING_BOILERPLATE_RE = re.compile(
    r"^(?:this\s+is\s+|there\s+is\s+)?"
    r"(?:a\s+|an\s+|the\s+)?"
    r"(?:close[- ]up\s+|detailed\s+|colorful\s+|vibrant\s+|traditional\s+|indian\s+|folk\s+)*"
    r"(?:madhubani\s+|mithila\s+)*"
    r"(?:folk\s+art\s+|folk\s+|tribal\s+)*"
    r"(?:painting|illustration|drawing|artwork|art\s+piece|art|image|picture|photo|photograph|mural|print|sketch|canvas|depiction|representation|scene)"
    r"(?:\s+(?:of|showing|depicting|featuring|with|that\s+shows))?\s+",
    re.IGNORECASE,
)

# Standalone style/medium words to strip anywhere in the caption so style stays in the trigger.
_STYLE_WORDS_RE = re.compile(
    r"\b(?:madhubani|mithila|folk\s+art|tribal\s+art|indian\s+art|"
    r"painting|paintings|painted|illustration|illustrated|drawing|drawn|"
    r"artwork|sketch|sketched|watercolor|acrylic|oil\s+on\s+canvas|canvas|mural)\b",
    re.IGNORECASE,
)

# Cleanup leftover articles/prepositions or double spaces after style-word removal.
_EMPTY_PREP_RE = re.compile(
    r"\b(?:in\s+a\s+style|in\s+the\s+style|in\s+a\s+traditional\s+style|of\s+a\s*,|in\s+a\s*,)\b",
    re.IGNORECASE,
)


def clean_style_caption(raw_caption: str, trigger_prefix: str = "in sks_mithila style") -> str:
    """Strip medium/style words from ``raw_caption`` and prepend ``trigger_prefix``."""
    text = raw_caption.strip()
    trigger_clean = trigger_prefix.strip().rstrip(",")

    # If the caption already starts with the trigger prefix, strip it first to normalize cleanly.
    if text.lower().startswith(trigger_clean.lower()):
        text = text[len(trigger_clean) :].lstrip(" ,.:;-")

    # Strip leading boilerplate like "The image shows a colorful painting of ..."
    text = re.sub(
        r"^(?:the\s+(?:image|picture|painting|illustration|artwork|drawing)\s+"
        r"(?:shows|depicts|features|is|portrays|presents)\s+)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = _LEADING_BOILERPLATE_RE.sub("", text)
    text = _STYLE_WORDS_RE.sub("", text)
    text = _EMPTY_PREP_RE.sub("", text)

    # Normalize punctuation and whitespace.
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"[,;]\s*[,;]+", ",", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,.;:-")

    if not text:
        text = "traditional motif with intricate geometric borders and floral patterns"

    # Lowercase the first character unless it's a proper noun we want to keep natural after comma.
    if len(text) > 1 and text[0].isupper() and text[1].islower():
        text = text[0].lower() + text[1:]

    return f"{trigger_clean}, {text}"


def load_florence2_captioner(
    model_id: str = "microsoft/Florence-2-base",
    task_prompt: str = "<DETAILED_CAPTION>",
    device: str | None = None,
) -> Callable[[Image.Image], str]:
    """Load Florence-2 (or BLIP fallback if requested) and return an image->text callable."""
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    def _caption(image: Image.Image) -> str:
        rgb = image.convert("RGB")
        inputs = processor(text=task_prompt, images=rgb, return_tensors="pt").to(device, dtype)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                num_beams=3,
                do_sample=False,
            )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(rgb.width, rgb.height),
        )
        if isinstance(parsed, dict) and task_prompt in parsed:
            return str(parsed[task_prompt]).strip()
        return str(parsed).strip()

    return _caption


def sync_metadata_from_txt(img_dir: Path) -> Path:
    """Build ``metadata.jsonl`` from existing ``.txt`` sidecars in ``img_dir``."""
    images = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
    meta_path = img_dir / "metadata.jsonl"
    lines: list[str] = []
    for img_path in images:
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        caption = txt_path.read_text(encoding="utf-8").strip()
        lines.append(json.dumps({"file_name": img_path.name, "text": caption}, ensure_ascii=False))
    meta_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return meta_path


def caption_dir(
    img_dir: Path,
    trigger_prefix: str = "in sks_mithila style",
    model_id: str = "microsoft/Florence-2-base",
    task_prompt: str = "<DETAILED_CAPTION>",
    captioner_fn: Callable[[Image.Image], str] | None = None,
    overwrite: bool = True,
) -> list[Path]:
    """Auto-caption all images in ``img_dir``, writing ``.txt`` files and ``metadata.jsonl``."""
    images = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
    if not images:
        return []

    if captioner_fn is None:
        captioner_fn = load_florence2_captioner(model_id=model_id, task_prompt=task_prompt)

    written_txts: list[Path] = []
    for img_path in images:
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists() and not overwrite:
            written_txts.append(txt_path)
            continue
        with Image.open(img_path) as im:
            raw = captioner_fn(im)
        cleaned = clean_style_caption(raw, trigger_prefix=trigger_prefix)
        txt_path.write_text(cleaned + "\n", encoding="utf-8")
        written_txts.append(txt_path)
        print(f"{img_path.name} -> {cleaned}")

    sync_metadata_from_txt(img_dir)
    return written_txts


if __name__ == "__main__":
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Auto-caption processed images for SDXL Style LoRA.")
    ap.add_argument("img_dir", type=Path, help="Directory of processed 1024x1024 images")
    ap.add_argument("--trigger", type=str, default=str(cfg.get("instance_prompt", "in sks_mithila style")), help="Trigger prefix")
    ap.add_argument("--model", type=str, default=str(cfg.get("captioner_model", "microsoft/Florence-2-base")), help="VLM model ID")
    ap.add_argument("--task", type=str, default=str(cfg.get("captioner_task", "<DETAILED_CAPTION>")), help="Florence-2 task prompt")
    ap.add_argument("--no-overwrite", action="store_true", help="Skip images that already have a .txt file")
    ap.add_argument("--sync-only", action="store_true", help="Rebuild metadata.jsonl from existing .txt files")
    a = ap.parse_args()

    if a.sync_only:
        out_meta = sync_metadata_from_txt(a.img_dir)
        print(f"synced {out_meta}")
    else:
        txts = caption_dir(
            a.img_dir,
            trigger_prefix=a.trigger,
            model_id=a.model,
            task_prompt=a.task,
            overwrite=not a.no_overwrite,
        )
        print(f"wrote {len(txts)} caption files and metadata.jsonl in {a.img_dir}")

"""Center-crop and resize training images to a square resolution (default 1024 for SDXL).

Usage: python -m lora_studio.prepare data/raw data/processed [--size 1024]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def center_crop_resize(img: Image.Image, size: int = 1024) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    return ImageOps.fit(img, (size, size), method=Image.LANCZOS, centering=(0.5, 0.5))


def prepare_dir(src: Path, dst: Path, size: int = 1024, min_side: int = 512) -> list[Path]:
    dst.mkdir(parents=True, exist_ok=True)
    out = []
    for i, p in enumerate(sorted(q for q in src.iterdir() if q.suffix.lower() in EXTS)):
        with Image.open(p) as im:
            if min(im.size) < min_side:
                print(f"skip (too small {im.size}): {p.name}")
                continue
            target = dst / f"{i:03d}.png"
            center_crop_resize(im, size).save(target)
            out.append(target)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--size", type=int, default=1024)
    a = ap.parse_args()
    print(f"wrote {len(prepare_dir(a.src, a.dst, a.size))} images to {a.dst}")

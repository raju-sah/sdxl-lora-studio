"""Center-crop, resize, deduplicate, and filter training images (default 1024x1024 for SDXL).

Usage: python -m lora_studio.prepare data/raw data/processed [--size 1024] [--min-sharpness 10.0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps, ImageStat

EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# 3x3 discrete Laplacian kernel shifted by +128 so signed responses fit in uint8 [0, 255].
_LAPLACIAN_KERNEL = ImageFilter.Kernel(
    size=(3, 3),
    kernel=[0, 1, 0, 1, -4, 1, 0, 1, 0],
    scale=1,
    offset=128,
)


def center_crop_resize(img: Image.Image, size: int = 1024) -> Image.Image:
    """EXIF-transpose, convert to RGB, and center-crop/resize to (size, size)."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    return ImageOps.fit(img, (size, size), method=Image.LANCZOS, centering=(0.5, 0.5))


def dhash(img: Image.Image, hash_size: int = 8) -> int:
    """Compute a 64-bit difference hash (dHash) using Pillow only."""
    gray = ImageOps.exif_transpose(img).convert("L").resize(
        (hash_size + 1, hash_size), Image.BILINEAR
    )
    pixels = gray.tobytes()
    w = hash_size + 1
    bits = 0
    for row in range(hash_size):
        row_offset = row * w
        for col in range(hash_size):
            bits = (bits << 1) | (1 if pixels[row_offset + col] > pixels[row_offset + col + 1] else 0)
    return bits


def hamming_distance(h1: int, h2: int) -> int:
    """Return bit count of XOR between two integer hashes."""
    return (h1 ^ h2).bit_count()


def sharpness_score(img: Image.Image, eval_size: int = 512) -> float:
    """Compute variance of the 3x3 Laplacian on a normalized grayscale thumbnail."""
    gray = ImageOps.exif_transpose(img).convert("L")
    gray.thumbnail((eval_size, eval_size), Image.BILINEAR)
    edges = gray.filter(_LAPLACIAN_KERNEL)
    return float(ImageStat.Stat(edges).var[0])


def prepare_dir(
    src: Path,
    dst: Path,
    size: int = 1024,
    min_side: int = 512,
    dedupe: bool = True,
    max_hamming: int = 6,
    min_sharpness: float = 0.0,
    max_images: int | None = None,
) -> list[Path]:
    """Filter, deduplicate, and center-crop images from ``src`` into ``dst`` as numbered PNGs."""
    dst.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    seen_hashes: list[tuple[int, str]] = []

    candidates = sorted(q for q in src.iterdir() if q.is_file() and q.suffix.lower() in EXTS)
    out_idx = 0
    for p in candidates:
        if max_images is not None and out_idx >= max_images:
            break
        with Image.open(p) as im:
            if min(im.size) < min_side:
                print(f"skip (too small {im.size}): {p.name}")
                continue
            if min_sharpness > 0.0:
                score = sharpness_score(im)
                if score < min_sharpness:
                    print(f"skip (blurry sharpness={score:.1f} < {min_sharpness}): {p.name}")
                    continue
            if dedupe:
                h = dhash(im)
                dup_of = next(
                    (prev_name for prev_h, prev_name in seen_hashes if hamming_distance(h, prev_h) <= max_hamming),
                    None,
                )
                if dup_of is not None:
                    print(f"skip (duplicate of {dup_of}): {p.name}")
                    continue
                seen_hashes.append((h, p.name))

            target = dst / f"{out_idx:03d}.png"
            center_crop_resize(im, size).save(target)
            out.append(target)
            out_idx += 1
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare SDXL LoRA training images.")
    ap.add_argument("src", type=Path, help="Raw images directory")
    ap.add_argument("dst", type=Path, help="Processed output directory")
    ap.add_argument("--size", type=int, default=1024, help="Target square resolution (default: 1024)")
    ap.add_argument("--min-side", type=int, default=512, help="Minimum shortest side in pixels")
    ap.add_argument("--min-sharpness", type=float, default=8.0, help="Minimum Laplacian variance sharpness")
    ap.add_argument("--max-hamming", type=int, default=6, help="Max dHash Hamming distance for duplicate detection")
    ap.add_argument("--max-images", type=int, default=30, help="Maximum number of images to output")
    ap.add_argument("--no-dedupe", action="store_true", help="Disable perceptual deduplication")
    a = ap.parse_args()
    written = prepare_dir(
        a.src,
        a.dst,
        size=a.size,
        min_side=a.min_side,
        dedupe=not a.no_dedupe,
        max_hamming=a.max_hamming,
        min_sharpness=a.min_sharpness,
        max_images=a.max_images,
    )
    print(f"wrote {len(written)} images to {a.dst}")

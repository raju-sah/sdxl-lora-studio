"""Center-crop, resize, trim studio margins/calibration bars, deduplicate, and filter training images.

Usage: python -m lora_studio.prepare data/raw data/processed [--size 1024] [--min-sharpness 8.0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps, ImageStat

from lora_studio.config import load_config

EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# 3x3 discrete Laplacian kernel shifted by +128 so signed responses fit in uint8 [0, 255].
_LAPLACIAN_KERNEL = ImageFilter.Kernel(
    size=(3, 3),
    kernel=[0, 1, 0, 1, -4, 1, 0, 1, 0],
    scale=1,
    offset=128,
)


def trim_studio_margins(
    img: Image.Image,
    max_margin_ratio: float = 0.25,
    std_threshold: float = 12.0,
    min_gutter_px: int = 12,
) -> Image.Image:
    """Detect and trim outer studio margins and museum color calibration bars.

    Museum scans (such as V&A IIIF items) often include an outer color calibration bar
    followed by a uniform light studio gutter before the actual painting border begins.
    Scanning inward up to ``max_margin_ratio`` on each side locates the innermost uniform
    light gutter (``stddev <= std_threshold`` and ``mean >= 205``) that starts near the edge
    (within 8% of side length) and crops cleanly to its inner boundary.
    """
    rgb = ImageOps.exif_transpose(img).convert("RGB")
    w, h = rgb.size
    if min(w, h) < 128 or max_margin_ratio <= 0.0:
        return rgb

    gray = rgb.convert("L")
    step = max(2, min(w, h) // 256)

    def _find_cut(length: int, is_vertical: bool, from_end: bool) -> int:
        limit = int(length * max_margin_ratio)
        quiet_runs: list[tuple[int, int]] = []
        run_start: int | None = None
        for offset in range(0, limit, step):
            pos = (length - offset - step) if from_end else offset
            box = (pos, 0, pos + step, h) if is_vertical else (0, pos, w, pos + step)
            st = ImageStat.Stat(gray.crop(box))
            mean_v, std_v = st.mean[0], st.stddev[0]
            if std_v <= std_threshold and mean_v >= 205.0:
                if run_start is None:
                    run_start = offset
            elif run_start is not None:
                if offset - run_start >= min_gutter_px:
                    quiet_runs.append((run_start, offset))
                run_start = None
        if run_start is not None and (limit - run_start) >= min_gutter_px:
            quiet_runs.append((run_start, limit))
        valid = [end for start, end in quiet_runs if start <= int(length * 0.08)]
        return valid[-1] if valid else 0

    left = _find_cut(w, is_vertical=True, from_end=False)
    right = w - _find_cut(w, is_vertical=True, from_end=True)
    top = _find_cut(h, is_vertical=False, from_end=False)
    bottom = h - _find_cut(h, is_vertical=False, from_end=True)

    if (
        (right - left) >= int(w * 0.5)
        and (bottom - top) >= int(h * 0.5)
        and (left > 0 or top > 0 or right < w or bottom < h)
    ):
        return rgb.crop((left, top, right, bottom))
    return rgb


def center_crop_resize(
    img: Image.Image,
    size: int = 1024,
    trim_margins: bool = True,
) -> Image.Image:
    """EXIF-transpose, convert to RGB, optionally trim studio margins, and center-crop/resize to (size, size)."""
    img = trim_studio_margins(img) if trim_margins else ImageOps.exif_transpose(img).convert("RGB")
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
    trim_margins: bool = True,
) -> list[Path]:
    """Filter, deduplicate, trim margins, and center-crop images from ``src`` into ``dst`` as numbered PNGs."""
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
            center_crop_resize(im, size=size, trim_margins=trim_margins).save(target)
            out.append(target)
            out_idx += 1
    return out


if __name__ == "__main__":
    cfg = load_config()
    ds_cfg = cfg.get("dataset", {})
    ap = argparse.ArgumentParser(description="Prepare SDXL LoRA training images.")
    ap.add_argument("src", type=Path, help="Raw images directory")
    ap.add_argument("dst", type=Path, help="Processed output directory")
    ap.add_argument("--size", type=int, default=int(cfg.get("resolution", 1024)), help="Target square resolution")
    ap.add_argument("--min-side", type=int, default=int(ds_cfg.get("min_side", 512)), help="Minimum shortest side in pixels")
    ap.add_argument("--min-sharpness", type=float, default=8.0, help="Minimum Laplacian variance sharpness")
    ap.add_argument("--max-hamming", type=int, default=6, help="Max dHash Hamming distance for duplicate detection")
    ap.add_argument("--max-images", type=int, default=int(ds_cfg.get("target_count", 30)), help="Maximum number of images to output")
    ap.add_argument("--no-dedupe", action="store_true", help="Disable perceptual deduplication")
    ap.add_argument("--no-trim-margins", action="store_true", help="Disable studio margin / calibration bar trimming")
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
        trim_margins=not a.no_trim_margins,
    )
    print(f"wrote {len(written)} images to {a.dst}")

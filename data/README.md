# Data (git-ignored)

- `data/raw/` — original images (15–40). Record source + rights here.
- `data/processed/` — 1024×1024 PNGs + one caption `.txt` per image (or `metadata.jsonl` with a `text` column).
- Upload: `kaggle datasets create -p data/processed` → `rajucode/my-lora-images` (private).

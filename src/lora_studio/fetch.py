"""Curate public-domain / Creative Commons / Open-Access Mithila (Madhubani) traditional folk
paintings from Wikimedia Commons and the Victoria & Albert Museum API into data/raw/ and log
full provenance in data/sources.csv.

Usage: python -m lora_studio.fetch [--raw-dir data/raw] [--manifest data/sources.csv] [--limit 30]
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

USER_AGENT = "SDXLLoRAStudio/0.1 (https://github.com/raju-sah/sdxl-lora-studio; educational research)"

COMMONS_CATEGORIES = [
    "Category:Madhubani painting",
    "Category:Art of Madhubani",
    "Category:Bharti Dayal",
    "Category:Mahasundari Devi",
]

COMMONS_SEARCH_QUERIES = [
    "Madhubani painting",
    "Mithila painting",
    "Madhubani folk art",
    "Mithila folk art",
    "Kohbar Madhubani",
]

# Reject filenames/titles that are photos of artists, stations, trains, buildings, shops, statues, or non-Mithila sketches.
_EXCLUDE_TITLE_RE = re.compile(
    r"(?:railway|station|junction|train|rail|temple|market|chowk|carnival|exhibition|salle|"
    r"exposition|shop|framed|artist|students|girls|women|appreciating|making|makes|"
    r"dupatta|bottle|certificate|banner|stamp|colnect|dilli\s*haat|restaurant|"
    r"kochi|koci|koch\b|kolkata|patna|darbhanga|khajuraho|little\s*world|aichi|"
    r"asha\s+jha|krishan\s+kumar\s+kashyap|statue|dadi5|img\s*0103|img\s*0028|"
    r"madhubani\s+painiting\.jpg|peace\s+in\s+mithila|mithila\s+painting\.jpg|"
    r"colorful\s+madhubani\s+painting\.jpg|modern\s+representation|file:madhubani\.jpg|"
    r"मैथिली\s+संस्कृति|on\s+body|on\s+face|ancient\s+folk|jadupatua|"
    r"file:madhubani\s+paintings\.jpg|devi6|devi4|jharkhand|a13r)",
    re.IGNORECASE,
)

_ALLOWED_LICENSE_RE = re.compile(
    r"(?:cc0|public\s+domain|pd|cc[- ]by|godl)",
    re.IGNORECASE,
)


def _strip_html(raw: str) -> str:
    clean = re.sub(r"<[^>]+>", "", raw or "")
    return re.sub(r"\s+", " ", html.unescape(clean)).strip()


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _query_commons(params: dict[str, str | int]) -> dict:
    query = {**params, "format": "json"}
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(query)
    return _fetch_json(url)


def _download_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return resp.read()


def discover_commons_candidates(min_side: int = 512) -> list[dict]:
    """Discover clean 2D Mithila/Madhubani artwork files on Wikimedia Commons."""
    titles: set[str] = set()
    for cat in COMMONS_CATEGORIES:
        res = _query_commons({"action": "query", "list": "categorymembers", "cmtitle": cat, "cmlimit": 200})
        for member in res.get("query", {}).get("categorymembers", []):
            if member.get("ns") == 6:
                titles.add(member["title"])

    for q in COMMONS_SEARCH_QUERIES:
        res = _query_commons(
            {
                "action": "query",
                "list": "search",
                "srsearch": f"filetype:bitmap {q}",
                "srnamespace": 6,
                "srlimit": 50,
            }
        )
        for item in res.get("query", {}).get("search", []):
            titles.add(item["title"])

    sorted_titles = sorted(titles)
    candidates: list[dict] = []

    for i in range(0, len(sorted_titles), 25):
        batch = sorted_titles[i : i + 25]
        info = _query_commons(
            {
                "action": "query",
                "titles": "|".join(batch),
                "prop": "imageinfo",
                "iiprop": "url|size|extmetadata|mime",
                "iiurlwidth": 1600,
            }
        )
        for page in info.get("query", {}).get("pages", {}).values():
            title = page.get("title", "")
            if not title or _EXCLUDE_TITLE_RE.search(title):
                continue
            ii = (page.get("imageinfo") or [None])[0]
            if not ii:
                continue
            mime = ii.get("mime", "")
            if mime not in ("image/jpeg", "image/png", "image/webp"):
                continue
            w, h = int(ii.get("width", 0)), int(ii.get("height", 0))
            if min(w, h) < min_side:
                continue
            aspect = max(w, h) / max(1, min(w, h))
            if aspect > 2.2:
                continue

            ext = ii.get("extmetadata", {})
            license_short = _strip_html(ext.get("LicenseShortName", {}).get("value", ""))
            if not _ALLOWED_LICENSE_RE.search(license_short):
                continue

            artist = _strip_html(ext.get("Artist", {}).get("value", "Unknown"))
            desc = _strip_html(ext.get("ImageDescription", {}).get("value", ""))

            download_url = ii.get("thumburl") if max(w, h) > 2048 and ii.get("thumburl") else ii["url"]
            candidates.append(
                {
                    "title": title.removeprefix("File:"),
                    "artist": artist or "Unknown",
                    "license": license_short,
                    "source_url": ii.get("descriptionurl", ii["url"]),
                    "download_url": download_url,
                    "width": w,
                    "height": h,
                    "description": desc[:240],
                }
            )

    candidates.sort(key=lambda c: (min(c["width"], c["height"]), c["title"]), reverse=True)
    return candidates


def discover_vam_candidates(max_records: int = 15) -> list[dict]:
    """Discover authentic Mithila/Madhubani paintings from the Victoria & Albert Museum IIIF API."""
    url = "https://api.vam.ac.uk/v2/objects/search?q=Madhubani&images_exist=1&page_size=50"
    data = _fetch_json(url)
    out: list[dict] = []
    for rec in data.get("records", []):
        if rec.get("objectType") not in ("Painting", "Drawing"):
            continue
        img_id = rec.get("_primaryImageId")
        sys_num = rec.get("systemNumber")
        if not img_id or not sys_num:
            continue
        title_raw = (rec.get("_primaryTitle") or "Madhubani Painting").strip()
        maker = (rec.get("_primaryMaker") or {}).get("name") or "Mithila Artist"
        out.append(
            {
                "title": f"VAM_{sys_num}_{title_raw}",
                "artist": maker,
                "license": "V&A Open Access (Non-Commercial / Educational)",
                "source_url": f"https://collections.vam.ac.uk/item/{sys_num}",
                "download_url": f"https://framemark.vam.ac.uk/collections/{img_id}/full/full/0/default.jpg",
                "width": 768,
                "height": 768,
                "crop_inset": 0.19,
                "description": f"{title_raw} — Mithila painting in Victoria & Albert Museum ({maker})",
            }
        )
        if len(out) >= max_records:
            break
    return out


def fetch_dataset(
    raw_dir: Path,
    manifest_path: Path,
    limit: int = 30,
    min_side: int = 512,
    clean_existing: bool = True,
) -> list[dict]:
    """Download up to ``limit`` curated Mithila/Madhubani paintings and write ``manifest_path``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if clean_existing:
        for old_file in raw_dir.glob("*.jpg"):
            old_file.unlink()
        for old_png in raw_dir.glob("*.png"):
            old_png.unlink()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    commons_cands = discover_commons_candidates(min_side=min_side)
    vam_cands = discover_vam_candidates(max_records=15)
    candidates = commons_cands + vam_cands
    print(f"discovered {len(commons_cands)} Commons + {len(vam_cands)} V&A Museum Mithila paintings")

    saved_rows: list[dict] = []
    for cand in candidates:
        if len(saved_rows) >= limit:
            break
        base_title = cand["title"].rsplit(".", 1)[0]
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", base_title).strip("_")[:48]
        filename = f"{len(saved_rows):03d}_{slug}.jpg"
        dest = raw_dir / filename

        try:
            raw_bytes = _download_bytes(cand["download_url"])
            with Image.open(io.BytesIO(raw_bytes)) as im:
                rgb = im.convert("RGB")
                inset = float(cand.get("crop_inset", 0.0))
                if inset > 0.0:
                    cw, ch = rgb.size
                    box = (int(cw * inset), int(ch * inset), int(cw * (1.0 - inset)), int(ch * (1.0 - inset)))
                    rgb = rgb.crop(box).resize((max(cw, min_side), max(ch, min_side)), Image.LANCZOS)
                if min(rgb.size) < min_side:
                    continue
                rgb.save(dest, format="JPEG", quality=95)
            with Image.open(dest) as check_im:
                actual_w, actual_h = check_im.size
        except (OSError, ValueError) as exc:
            print(f"skip ({cand['title']}): {exc}")
            if dest.exists():
                dest.unlink()
            continue

        row = {
            "filename": filename,
            "title": cand["title"],
            "artist": cand["artist"],
            "license": cand["license"],
            "source_url": cand["source_url"],
            "width": actual_w,
            "height": actual_h,
            "description": cand["description"],
        }
        saved_rows.append(row)
        print(f"[{len(saved_rows):02d}/{limit}] {filename} ({actual_w}x{actual_h}, {cand['license']})")

    fieldnames = ["filename", "title", "artist", "license", "source_url", "width", "height", "description"]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(saved_rows)

    print(f"wrote {len(saved_rows)} records to {manifest_path}")
    return saved_rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch public-domain / CC Mithila (Madhubani) paintings.")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory to store raw images")
    ap.add_argument("--manifest", type=Path, default=Path("data/sources.csv"), help="CSV license manifest")
    ap.add_argument("--limit", type=int, default=30, help="Max raw artworks to download")
    ap.add_argument("--min-side", type=int, default=512, help="Minimum shortest side in pixels")
    a = ap.parse_args()
    fetch_dataset(a.raw_dir, a.manifest, limit=a.limit, min_side=a.min_side)

import json

from PIL import Image

from lora_studio.caption import caption_dir, clean_style_caption, sync_metadata_from_txt


def test_clean_style_caption_strips_medium_and_style_words():
    raw = "The image shows a colorful traditional Madhubani folk art painting of two peacocks perched on a lotus branch."
    cleaned = clean_style_caption(raw, trigger_prefix="in sks_mithila style")
    assert cleaned.startswith("in sks_mithila style, ")
    assert "madhubani" not in cleaned.lower()
    assert "painting" not in cleaned.lower()
    assert "folk art" not in cleaned.lower()
    assert "two peacocks perched on a lotus branch" in cleaned.lower()


def test_caption_dir_and_metadata_jsonl(tmp_path):
    Image.new("RGB", (512, 512), (200, 50, 50)).save(tmp_path / "000.png")
    Image.new("RGB", (512, 512), (50, 50, 200)).save(tmp_path / "001.png")

    responses = iter(
        [
            "A vibrant Mithila illustration of a golden fish surrounded by floral borders.",
            "An Indian folk art drawing showing an elephant and musicians in a marriage procession.",
        ]
    )

    txts = caption_dir(
        tmp_path,
        trigger_prefix="in sks_mithila style",
        captioner_fn=lambda _im: next(responses),
    )
    assert len(txts) == 2
    assert (tmp_path / "000.txt").exists()
    assert (tmp_path / "001.txt").exists()

    meta_path = tmp_path / "metadata.jsonl"
    assert meta_path.exists()
    rows = [json.loads(line) for line in meta_path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["file_name"] == "000.png"
    assert rows[0]["text"].startswith("in sks_mithila style, ")
    assert "golden fish" in rows[0]["text"]

    # Hand-edit 000.txt and verify sync_metadata_from_txt updates metadata.jsonl
    (tmp_path / "000.txt").write_text("in sks_mithila style, hand-edited lotus motif\n")
    sync_metadata_from_txt(tmp_path)
    updated_rows = [json.loads(line) for line in meta_path.read_text().splitlines()]
    assert updated_rows[0]["text"] == "in sks_mithila style, hand-edited lotus motif"

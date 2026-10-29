from PIL import Image

from lora_studio.config import load_config
from lora_studio.fetch import _strip_html
from lora_studio.grid import (
    load_prompts,
    restitch_existing_grid,
    stitch_checkpoint_grid,
    wrap_prompt_label,
)
from lora_studio.metrics import (
    format_metrics_markdown,
    select_best_checkpoint,
    strip_trigger_prefix,
)


def test_strip_html():
    assert _strip_html("<b>CC BY-SA 4.0</b> &amp; Public Domain") == "CC BY-SA 4.0 & Public Domain"


def test_load_prompts_and_stitch_grid(tmp_path):
    pfile = tmp_path / "prompts.txt"
    pfile.write_text(
        "# comment\nin sks_mithila style, two peacocks\nin sks_mithila style, astronaut on the moon\n"
    )
    prompts = load_prompts(pfile)
    assert len(prompts) == 2

    grid_imgs = {
        "Base SDXL": [Image.new("RGB", (128, 128), (100, 0, 0)), Image.new("RGB", (128, 128), (0, 100, 0))],
        "ckpt-1000": [Image.new("RGB", (128, 128), (0, 0, 100)), Image.new("RGB", (128, 128), (100, 100, 0))],
    }
    canvas = stitch_checkpoint_grid(grid_imgs, prompts, tile_size=64, label_width=100, header_height=30)
    assert canvas.size == (100 + 2 * 64, 30 + 2 * 64)


def test_strip_trigger_and_select_best_checkpoint():
    assert (
        strip_trigger_prefix("in sks_mithila style, an astronaut riding a horse")
        == "an astronaut riding a horse"
    )

    mock_results = [
        {"checkpoint": 0, "clip_content_score": 0.315, "dino_style_similarity": 0.240, "dino_max_nn": 0.35},
        {"checkpoint": 250, "clip_content_score": 0.312, "dino_style_similarity": 0.410, "dino_max_nn": 0.52},
        {"checkpoint": 500, "clip_content_score": 0.310, "dino_style_similarity": 0.510, "dino_max_nn": 0.61},
        {"checkpoint": 750, "clip_content_score": 0.308, "dino_style_similarity": 0.575, "dino_max_nn": 0.68},
        {"checkpoint": 1000, "clip_content_score": 0.305, "dino_style_similarity": 0.630, "dino_max_nn": 0.74},
        {"checkpoint": 1250, "clip_content_score": 0.292, "dino_style_similarity": 0.655, "dino_max_nn": 0.81},
        {"checkpoint": 1500, "clip_content_score": 0.265, "dino_style_similarity": 0.680, "dino_max_nn": 0.91},
    ]
    best = select_best_checkpoint(mock_results, clip_drop_tolerance=0.015)
    assert best["checkpoint"] == 1000

    md = format_metrics_markdown(mock_results, best_ckpt=1000)
    assert "ckpt-1000" in md
    assert "Selected Best" in md


def test_config_and_restitch_grid(tmp_path):
    cfg = load_config()
    assert cfg["trigger"] == "sks_mithila"
    assert cfg["rank"] == 32
    assert cfg["hosting"]["hf_space_id"] == "raju-ai/sdxl-mithila-lora-demo"

    wrapped = wrap_prompt_label(
        "in sks_mithila style, two peacocks perched on a flowering kadamba tree branch",
        row_idx=0,
    )
    assert wrapped.startswith("P1: two peacocks")
    assert "\n" in wrapped

    prompts = ["in sks_mithila style, two peacocks", "in sks_mithila style, astronaut on the moon"]
    grid_imgs = {
        "Base SDXL": [Image.new("RGB", (64, 64), (120, 10, 10)), Image.new("RGB", (64, 64), (10, 120, 10))],
        "ckpt-250": [Image.new("RGB", (64, 64), (10, 10, 120)), Image.new("RGB", (64, 64), (120, 120, 10))],
    }
    grid_path = tmp_path / "checkpoint_grid.png"
    stitch_checkpoint_grid(grid_imgs, prompts, tile_size=64, label_width=100, header_height=30).save(grid_path)

    restitched = restitch_existing_grid(
        grid_path,
        prompts=prompts,
        checkpoints=[0, 250],
        tile_size=64,
        label_width=100,
        header_height=30,
    )
    assert restitched.size == (100 + 2 * 64, 30 + 2 * 64)
    # Verify tile pixel colors were preserved losslessly
    assert restitched.getpixel((100 + 32, 30 + 32)) == (120, 10, 10)
    assert restitched.getpixel((100 + 64 + 32, 30 + 64 + 32)) == (120, 120, 10)


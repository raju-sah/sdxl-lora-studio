from PIL import Image, ImageDraw

from lora_studio.prepare import (
    center_crop_resize,
    dhash,
    hamming_distance,
    prepare_dir,
    sharpness_score,
)


def _make_patterned_image(seed: int, size: tuple[int, int] = (800, 600)) -> Image.Image:
    img = Image.new("RGB", size, (240, 230, 200))
    draw = ImageDraw.Draw(img)
    for i in range(0, size[0], 32):
        color = ((i * (seed + 3)) % 255, ((i + seed * 47) * 7) % 255, (seed * 91) % 255)
        draw.rectangle([i, (i * seed) % (size[1] // 2), i + 16, size[1] - 20], fill=color, outline=(0, 0, 0))
    return img


def test_center_crop_resize():
    out = center_crop_resize(Image.new("RGB", (1600, 900)), 1024)
    assert out.size == (1024, 1024)


def test_prepare_dir(tmp_path):
    src, dst = tmp_path / "raw", tmp_path / "out"
    src.mkdir()
    Image.new("RGB", (800, 600)).save(src / "a.jpg")
    Image.new("RGB", (100, 100)).save(src / "tiny.png")  # skipped
    assert len(prepare_dir(src, dst, size=256)) == 1


def test_prepare_dir_dedupe_and_sharpness(tmp_path):
    src, dst = tmp_path / "raw", tmp_path / "out"
    src.mkdir()

    sharp_a = _make_patterned_image(1)
    sharp_a_dup = sharp_a.copy()
    sharp_b = _make_patterned_image(9)
    flat_blurry = Image.new("RGB", (800, 600), (128, 128, 128))

    sharp_a.save(src / "01_sharp_a.png")
    sharp_a_dup.save(src / "02_sharp_a_dup.png")
    sharp_b.save(src / "03_sharp_b.png")
    flat_blurry.save(src / "04_blurry.png")

    assert hamming_distance(dhash(sharp_a), dhash(sharp_a_dup)) == 0
    assert sharpness_score(sharp_a) > 10.0
    assert sharpness_score(flat_blurry) == 0.0

    out = prepare_dir(src, dst, size=256, dedupe=True, min_sharpness=5.0)
    assert len(out) == 2
    assert [p.name for p in out] == ["000.png", "001.png"]

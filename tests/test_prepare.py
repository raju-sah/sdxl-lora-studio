from PIL import Image

from lora_studio.prepare import center_crop_resize, prepare_dir


def test_center_crop_resize():
    out = center_crop_resize(Image.new("RGB", (1600, 900)), 1024)
    assert out.size == (1024, 1024)


def test_prepare_dir(tmp_path):
    src, dst = tmp_path / "raw", tmp_path / "out"
    src.mkdir()
    Image.new("RGB", (800, 600)).save(src / "a.jpg")
    Image.new("RGB", (100, 100)).save(src / "tiny.png")  # skipped
    assert len(prepare_dir(src, dst, size=256)) == 1

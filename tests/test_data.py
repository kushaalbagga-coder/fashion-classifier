from pathlib import Path

import pytest
from PIL import Image

from src.data import build_dataloaders


def _add_image(root: Path, split: str, category: str) -> None:
    folder = root / split / category
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12)).save(folder / "sample.png")


def test_builds_loaders_and_checks_classes(tmp_path: Path) -> None:
    for split in ("train", "val"):
        for category in ("dress", "shoe"):
            _add_image(tmp_path, split, category)
    train_loader, val_loader, classes = build_dataloaders(tmp_path, 16, 2, 0)
    assert classes == ["dress", "shoe"]
    assert next(iter(train_loader))[0].shape == (2, 3, 16, 16)
    assert next(iter(val_loader))[0].shape == (2, 3, 16, 16)


def test_rejects_different_split_classes(tmp_path: Path) -> None:
    _add_image(tmp_path, "train", "dress")
    _add_image(tmp_path, "val", "shoe")
    with pytest.raises(ValueError, match="identical class folders"):
        build_dataloaders(tmp_path, 16, 2, 0)

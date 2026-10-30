from pathlib import Path

import pytest

from src.config import Config


def test_loads_config() -> None:
    config = Config.from_yaml(Path("configs/config.yaml"))
    config.validate()
    assert config.model_name == "efficientnet_b0"
    assert config.best_model == Path("models/best_model.pth")


def test_rejects_invalid_batch_size(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("data:\n  batch_size: 0\n", encoding="utf-8")
    config = Config.from_yaml(path)
    with pytest.raises(ValueError, match="batch_size"):
        config.validate()

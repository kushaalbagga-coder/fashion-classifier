from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    data_root: Path
    image_size: int
    batch_size: int
    num_workers: int
    model_name: str
    dropout: float
    epochs: int
    finetune_after_epoch: int
    lr_head: float
    lr_finetune: float
    weight_decay: float
    device: str
    best_model: Path

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with Path(path).open(encoding="utf-8") as file:
            raw: dict[str, Any] = yaml.safe_load(file) or {}

        data = raw.get("data", {})
        model = raw.get("model", {})
        training = raw.get("training", {})
        return cls(
            data_root=Path(data.get("root", "data")),
            image_size=int(data.get("image_size", 224)),
            batch_size=int(data.get("batch_size", 32)),
            num_workers=int(data.get("num_workers", 2)),
            model_name=model.get("name", "efficientnet_b0"),
            dropout=float(model.get("dropout", 0.3)),
            epochs=int(training.get("epochs", 15)),
            finetune_after_epoch=int(training.get("finetune_after_epoch", 5)),
            lr_head=float(training.get("lr_head", 1e-3)),
            lr_finetune=float(training.get("lr_finetune", 5e-5)),
            weight_decay=float(training.get("weight_decay", 1e-4)),
            device=training.get("device", "auto"),
            best_model=Path(training.get("best_model", "models/best_model.pth")),
        )

    def validate(self) -> None:
        if self.image_size < 1:
            raise ValueError("data.image_size must be greater than zero")
        if self.batch_size < 1:
            raise ValueError("data.batch_size must be greater than zero")
        if self.num_workers < 0:
            raise ValueError("data.num_workers cannot be negative")
        if self.epochs < 1:
            raise ValueError("training.epochs must be greater than zero")
        if not 0 <= self.finetune_after_epoch <= self.epochs:
            raise ValueError("training.finetune_after_epoch must be between zero and training.epochs")
        if self.dropout < 0:
            raise ValueError("model.dropout cannot be negative")
        if self.lr_head <= 0 or self.lr_finetune <= 0:
            raise ValueError("Training learning rates must be greater than zero")
        if self.device != "auto" and self.device not in {"cpu", "cuda"}:
            raise ValueError("training.device must be 'auto', 'cpu', or 'cuda'")

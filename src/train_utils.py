import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.model import unfreeze_backbone


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += labels.size(0)
    if count == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    return total_loss / count, correct / count


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    best_path: str | Path,
    class_names: list[str],
    model_name: str,
    image_size: int,
    finetune_after_epoch: int,
    lr_finetune: float,
    weight_decay: float,
) -> dict[str, Any]:
    best_accuracy = -1.0
    history: dict[str, Any] = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_path = Path(best_path)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    for epoch in range(epochs):
        if epoch == finetune_after_epoch:
            unfreeze_backbone(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr_finetune, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(epochs - epoch, 1)
            )

        started = time.monotonic()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.2%} elapsed={time.monotonic() - started:.0f}s"
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": model_name,
                    "class_names": class_names,
                    "image_size": image_size,
                    "val_accuracy": val_accuracy,
                },
                best_path,
            )
            print(f"Saved best checkpoint ({best_accuracy:.2%}) to {best_path}")

    history["best_acc"] = best_accuracy
    return history

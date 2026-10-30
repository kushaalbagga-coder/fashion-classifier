import argparse

import torch
from torch import nn

from src.config import Config
from src.data import build_dataloaders
from src.model import build_model, freeze_backbone
from src.train_utils import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fashion image classifier")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    config = Config.from_yaml(args.config)
    config.validate()

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(
        "cuda" if config.device == "auto" and torch.cuda.is_available() else
        "cpu" if config.device == "auto" else config.device
    )
    train_loader, val_loader, class_names = build_dataloaders(
        config.data_root, config.image_size, config.batch_size, config.num_workers
    )
    print(f"Training on {device}; classes: {class_names}")
    model = build_model(config.model_name, len(class_names), config.dropout).to(device)
    freeze_backbone(model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.lr_head,
        weight_decay=config.weight_decay,
    )
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=device,
        epochs=config.epochs,
        best_path=config.best_model,
        class_names=class_names,
        model_name=config.model_name,
        image_size=config.image_size,
        finetune_after_epoch=config.finetune_after_epoch,
        lr_finetune=config.lr_finetune,
        weight_decay=config.weight_decay,
    )
    print(f"Best validation accuracy: {history['best_acc']:.2%}")


if __name__ == "__main__":
    main()

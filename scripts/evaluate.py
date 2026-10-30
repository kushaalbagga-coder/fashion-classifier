import argparse

import torch
from sklearn.metrics import classification_report

from src.config import Config
from src.data import build_dataloaders
from src.predict import FashionClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test split")
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
    _, _, test_loader, class_names = build_dataloaders(
        config.data_root,
        config.image_size,
        config.batch_size,
        config.num_workers,
        include_test=True,
    )
    classifier = FashionClassifier(config.best_model, device=device)
    if classifier.class_names != class_names:
        raise ValueError("Checkpoint class labels do not match the dataset")

    actual: list[int] = []
    predicted: list[int] = []
    with torch.inference_mode():
        for images, labels in test_loader:
            outputs = classifier.model(images.to(device))
            predicted.extend(outputs.argmax(dim=1).cpu().tolist())
            actual.extend(labels.tolist())
    print(classification_report(actual, predicted, target_names=class_names, zero_division=0))


if __name__ == "__main__":
    main()

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(round(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_dataloaders(
    root: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    include_test: bool = False,
) -> tuple[DataLoader, DataLoader, list[str]] | tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    root = Path(root)
    train_dataset = datasets.ImageFolder(root / "train", transform=get_train_transforms(image_size))
    val_dataset = datasets.ImageFolder(root / "val", transform=get_eval_transforms(image_size))
    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Training and validation splits must contain identical class folders")

    options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **options)
    val_loader = DataLoader(val_dataset, shuffle=False, **options)
    if not include_test:
        return train_loader, val_loader, train_dataset.classes

    test_dataset = datasets.ImageFolder(root / "test", transform=get_eval_transforms(image_size))
    if train_dataset.classes != test_dataset.classes:
        raise ValueError("Training and test splits must contain identical class folders")
    test_loader = DataLoader(test_dataset, shuffle=False, **options)
    return train_loader, val_loader, test_loader, train_dataset.classes

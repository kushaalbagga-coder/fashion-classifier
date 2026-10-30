from torch import nn
from torchvision import models


def build_model(
    name: str = "efficientnet_b0",
    num_classes: int = 8,
    dropout: float = 0.3,
    pretrained: bool = True,
) -> nn.Module:
    if num_classes < 1:
        raise ValueError("num_classes must be greater than zero")
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.classifier[1].in_features, num_classes),
        )
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    else:
        raise ValueError(f"Unsupported model architecture: {name}")
    return model


def freeze_backbone(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "classifier" in name or name.startswith("fc.")


def unfreeze_backbone(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True

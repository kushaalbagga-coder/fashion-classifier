from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from PIL import Image
from torchvision import transforms

from src.data import IMAGENET_MEAN, IMAGENET_STD
from src.model import build_model

DEFAULT_CLASSES = ["Tshirts", "Shirts", "Jeans", "Dresses", "Jackets", "Shoes", "Bags", "Watches"]


class FashionClassifier:
    def __init__(
        self,
        path: str | Path,
        class_names: list[str] | None = None,
        size: int | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint: Any = torch.load(path, map_location=self.device, weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must contain a model state dictionary")

        self.class_names = class_names or checkpoint.get("class_names") or DEFAULT_CLASSES
        self.model_name = checkpoint.get("model_name", "efficientnet_b0")
        self.val_accuracy = checkpoint.get("val_accuracy")
        image_size = size or checkpoint.get("image_size", 224)
        state = checkpoint.get("model_state", checkpoint)
        self.model = build_model(
            self.model_name, num_classes=len(self.class_names), pretrained=False
        ).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize(round(image_size * 1.14)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 3) -> dict[str, Any]:
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probabilities = functional.softmax(self.model(tensor), dim=1)[0]
        scores, indexes = probabilities.topk(min(top_k, len(self.class_names)))
        results = [
            {"label": self.class_names[index], "score": round(float(score), 4)}
            for score, index in zip(scores, indexes)
        ]
        return {
            "category": results[0]["label"],
            "confidence": results[0]["score"],
            "top_k": results,
        }

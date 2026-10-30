from pathlib import Path

import torch
from PIL import Image

from src.model import build_model
from src.predict import FashionClassifier


def test_checkpoint_metadata_and_prediction(tmp_path: Path) -> None:
    path = tmp_path / "model.pth"
    model = build_model("efficientnet_b0", num_classes=2, pretrained=False)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": "efficientnet_b0",
            "class_names": ["dress", "shoe"],
            "image_size": 64,
        },
        path,
    )
    classifier = FashionClassifier(path, device="cpu")
    result = classifier.predict(Image.new("RGB", (80, 80)), top_k=2)
    assert classifier.class_names == ["dress", "shoe"]
    assert result["category"] in classifier.class_names
    assert len(result["top_k"]) == 2
    assert sum(item["score"] for item in result["top_k"]) == 1.0

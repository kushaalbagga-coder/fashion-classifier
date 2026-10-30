import io
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.model import build_model


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    checkpoint = tmp_path / "model.pth"
    model = build_model("efficientnet_b0", num_classes=2, pretrained=False)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": "efficientnet_b0",
            "class_names": ["dress", "shoe"],
            "image_size": 64,
        },
        checkpoint,
    )
    monkeypatch.setattr("src.api.MODEL_PATH", checkpoint)
    from src.api import app

    with TestClient(app) as test_client:
        yield test_client


def test_health_and_class_endpoints(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/classes").json() == {"classes": ["dress", "shoe"]}


def test_predicts_valid_image(client: TestClient) -> None:
    image_data = io.BytesIO()
    Image.new("RGB", (64, 64)).save(image_data, format="JPEG")
    response = client.post(
        "/predict",
        files={"file": ("sample.jpg", image_data.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["category"] in {"dress", "shoe"}


def test_rejects_unsupported_media_type(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("sample.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415

import torch

from src.model import build_model, freeze_backbone, unfreeze_backbone


def test_model_output_and_freeze_cycle() -> None:
    model = build_model("efficientnet_b0", num_classes=3, pretrained=False)
    freeze_backbone(model)
    assert all(
        parameter.requires_grad == ("classifier" in name)
        for name, parameter in model.named_parameters()
    )
    unfreeze_backbone(model)
    assert all(parameter.requires_grad for parameter in model.parameters())
    model.eval()
    with torch.inference_mode():
        assert model(torch.randn(1, 3, 64, 64)).shape == (1, 3)

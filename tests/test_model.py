"""Shape and interface tests for the tactile modulus model."""

import torch

from nn_modules_resdeformabletrans import PhysiNet


def make_model() -> PhysiNet:
    model = PhysiNet(
        in_channels=3,
        time_steps=3,
        image_embed_dim=32,
        decoder_hidden_dims=(64, 32),
        pretrained_backbone=False,
    )
    model.eval()
    return model


def test_sequence_input_forward_shape() -> None:
    model = make_model()
    images = torch.randn(2, 3, 3, 64, 64)
    forces = torch.rand(2, 3)
    widths = torch.zeros(2, 3)
    with torch.no_grad():
        predictions = model(images, forces, widths)
    assert predictions.shape == (2,)
    assert torch.isfinite(predictions).all()


def test_flattened_legacy_input_forward_shape() -> None:
    model = make_model()
    images = torch.randn(2, 9, 64, 64)
    forces = torch.rand(2, 3)
    widths = torch.zeros(2, 3)
    with torch.no_grad():
        predictions = model(images, forces, widths)
    assert predictions.shape == (2,)

"""Unit tests: model definition (src/train/model.py)."""

import pytest
import torch

from src.train.model import LeafNet


@pytest.mark.parametrize("backbone", ["mobilenet_v2", "resnet18"])
def test_forward_shape(backbone):
    m = LeafNet(10, backbone, pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.rand(2, 3, 128, 128))
    assert out.shape == (2, 10)


def test_freeze_backbone_only_head_trainable():
    m = LeafNet(5, "mobilenet_v2", pretrained=False, freeze_backbone=True)
    assert not any(p.requires_grad for p in m.features.parameters())
    assert all(p.requires_grad for p in m.head.parameters())


def test_unfrozen_backbone_trainable():
    m = LeafNet(5, "mobilenet_v2", pretrained=False, freeze_backbone=False)
    assert any(p.requires_grad for p in m.features.parameters())


def test_normalisation_is_inside_model():
    m = LeafNet(3, "mobilenet_v2", pretrained=False)
    assert torch.allclose(m.mean.flatten(), torch.tensor([0.485, 0.456, 0.406]))


def test_unknown_backbone():
    with pytest.raises(ValueError):
        LeafNet(3, "vgg99", pretrained=False)

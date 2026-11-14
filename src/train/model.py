"""Model definition shared by training, evaluation and the serving backend."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LeafNet(nn.Module):
    """Transfer-learning classifier.

    Input : float32 tensor [N, 3, H, W] with pixel values in [0, 1]
    Output: logits [N, num_classes]

    ImageNet normalisation is done *inside* the model so any client (FastAPI,
    `mlflow models serve`, tests) only needs to send scaled pixels.
    """

    def __init__(self, num_classes: int, backbone: str = "mobilenet_v2", pretrained: bool = True,
                 freeze_backbone: bool = True):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "mobilenet_v2":
            net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
            net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
            self.features, self.head = net.features, net.classifier
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.features = nn.Sequential(*list(net.children())[:-2])
            self.head = nn.Linear(net.fc.in_features, num_classes)
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(f"unknown backbone {backbone!r}")

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        fmap = self.features(x)  # last conv feature map (used by Grad-CAM)
        return self.head(torch.flatten(self.pool(fmap), 1))

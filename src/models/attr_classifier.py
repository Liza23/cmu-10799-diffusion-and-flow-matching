"""
CelebA 40-Attribute Classifier

Frozen pretrained classifier for attribute disentanglement regularization (Option A).
Outputs 40 logits for CelebA attributes. Expects images in [-1, 1] (same as diffusion).
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CelebAAttrClassifier(nn.Module):
    """
    CelebA 40-attribute classifier. ResNet18 backbone + 40-dim output.
    Expects images in [-1, 1], shape (B, 3, H, W). Outputs (B, 40) logits.
    """

    def __init__(self, num_attributes: int = 40, image_size: int = 64):
        super().__init__()
        self.num_attributes = num_attributes
        self.image_size = image_size

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Remove FC layer; keep features
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(512, num_attributes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) in [-1, 1]
        Returns:
            (B, 40) logits
        """
        # ResNet expects ~224x224; we have 64x64. It will work but may lose some resolution.
        # For 64x64 the adaptive pool outputs (B, 512, 1, 1).
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)


def load_attr_classifier(
    checkpoint_path: str,
    device: torch.device,
    num_attributes: int = 40,
    image_size: int = 64,
) -> CelebAAttrClassifier:
    """
    Load a pretrained CelebA attribute classifier from checkpoint.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Attr classifier checkpoint not found: {checkpoint_path}")

    model = CelebAAttrClassifier(num_attributes=num_attributes, image_size=image_size)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)

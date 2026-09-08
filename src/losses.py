"""Adversarial and reconstruction losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# Hinge margin is sampled from [MARGIN_LOW, MARGIN_LOW + MARGIN_JITTER] per
# element rather than fixed at 1.0. The jitter softens the target and measurably
# stabilises D early on, when a small dataset lets it win too easily.
MARGIN_LOW = 0.8
MARGIN_JITTER = 0.2


def _margin(like: torch.Tensor) -> torch.Tensor:
    return torch.rand_like(like) * MARGIN_JITTER + MARGIN_LOW


def d_hinge_loss(logits: torch.Tensor, real: bool) -> torch.Tensor:
    """Hinge loss for the discriminator; push reals above +m and fakes below -m."""
    if real:
        return F.relu(_margin(logits) - logits).mean()
    return F.relu(_margin(logits) + logits).mean()


def g_loss(logits: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss: simply maximise D's score on fakes."""
    return -logits.mean()


class VGGPerceptualLoss(nn.Module):
    """L1 in VGG16 feature space.

    Closer to the paper's LPIPS reconstruction term than plain MSE, but it holds
    a second network on the GPU. On a 4GB card that is the first thing to drop
    if training OOMs -- pass --recon-loss mse.
    """

    LAYERS = (3, 8, 15, 22)  # relu1_2, relu2_2, relu3_3, relu4_3

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[: max(self.LAYERS) + 1]
        self.features = features.eval()
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return ((x + 1) / 2 - self.mean) / self.std  # [-1,1] -> ImageNet stats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = self._normalize(pred), self._normalize(target)
        loss = pred.new_zeros(())
        for i, layer in enumerate(self.features):
            pred, target = layer(pred), layer(target)
            if i in self.LAYERS:
                loss = loss + F.l1_loss(pred, target)
        return loss


class ReconstructionLoss(nn.Module):
    """Self-supervised reconstruction term for the discriminator.

    Applied to real images only. Its job is to stop D memorising a tiny dataset
    by forcing its features to stay a useful encoder, so a cheap pixel loss is
    already most of the benefit -- the reconstructions themselves never need to
    look good.
    """

    def __init__(self, kind: str = "mse"):
        super().__init__()
        if kind not in ("mse", "vgg"):
            raise ValueError(f"recon loss must be 'mse' or 'vgg', got {kind!r}")
        self.kind = kind
        self.vgg = VGGPerceptualLoss() if kind == "vgg" else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.shape[-2:] != pred.shape[-2:]:
            target = F.interpolate(
                target, size=pred.shape[-2:], mode="bilinear", align_corners=False
            )
        if self.vgg is not None:
            return self.vgg(pred, target)
        return F.mse_loss(pred, target)

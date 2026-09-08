"""Shared building blocks for the FastGAN generator and discriminator.

Follows Liu et al., ICLR 2021, "Towards Faster and Stabilized GAN Training for
High-fidelity Few-shot Image Synthesis". Spectral norm on every conv, GLU rather
than ReLU in the generator, and nearest-neighbour upsampling rather than
transposed convs above 4x4 (which avoids checkerboard artefacts).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


def conv2d(*args, **kwargs) -> nn.Module:
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def conv_transpose2d(*args, **kwargs) -> nn.Module:
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))


class GLU(nn.Module):
    """Gated linear unit over channels: halves the channel count.

    Every layer that feeds a GLU therefore has to produce 2x the channels it
    wants out -- that doubling is not a typo where it appears below.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.size(1)
        if channels % 2 != 0:
            raise ValueError(f"GLU needs an even channel count, got {channels}")
        half = channels // 2
        return x[:, :half] * torch.sigmoid(x[:, half:])


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class NoiseInjection(nn.Module):
    """Per-pixel noise with a learned, zero-initialised scale.

    Starts as a no-op, so the model only uses it if it helps.

    Set `deterministic` to drop the noise term and return the expected output.
    Since the noise is zero-mean, E[x + w*noise] == x exactly, so this is the
    mean prediction rather than an approximation. Serving needs it for two
    reasons: a fresh draw per forward would make the same seed render a
    different image every time (destroying the permalink and cache design), and
    the ONNX fp16 converter cannot handle the RandomNormalLike node it emits.
    """

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.deterministic = False

    def forward(self, x: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if self.deterministic:
            return x
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight * noise


def set_deterministic_noise(model: nn.Module, enabled: bool = True) -> int:
    """Toggle every NoiseInjection in a model; returns how many were changed."""
    count = 0
    for module in model.modules():
        if isinstance(module, NoiseInjection):
            module.deterministic = enabled
            count += 1
    return count


class InitLayer(nn.Module):
    """z -> 4x4 feature map."""

    def __init__(self, z_dim: int, channels: int):
        super().__init__()
        self.main = nn.Sequential(
            conv_transpose2d(z_dim, channels * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(channels * 2),
            GLU(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z.view(z.size(0), -1, 1, 1))


class UpBlock(nn.Module):
    """2x nearest upsample + 3x3 conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv2d(in_ch, out_ch * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch * 2),
            GLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class UpBlockComp(nn.Module):
    """Heavier upsample block: two convs plus noise injection."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = conv2d(in_ch, out_ch * 2, 3, 1, 1, bias=False)
        self.noise1 = NoiseInjection()
        self.norm1 = nn.BatchNorm2d(out_ch * 2)
        self.act1 = GLU()
        self.conv2 = conv2d(out_ch, out_ch * 2, 3, 1, 1, bias=False)
        self.noise2 = NoiseInjection()
        self.norm2 = nn.BatchNorm2d(out_ch * 2)
        self.act2 = GLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.noise1(self.conv1(self.up(x)))))
        return self.act2(self.norm2(self.noise2(self.conv2(x))))


class SkipLayerExcitation(nn.Module):
    """Channel-wise gate carried from a low-res feature map to a high-res one.

    This is the load-bearing idea of the paper. It gives the generator
    long-range conditioning -- 8x8 structure directly modulating 128x128
    detail -- for the cost of a 4x4 pooled conv, which is what keeps the whole
    model inside 4GB. Spatial size of `low` is irrelevant, since it is pooled
    to 4x4 first.
    """

    POOLED = 4  # every SLE reduces its low-res input to 4x4 before gating

    def __init__(self, low_ch: int, high_ch: int, low_size: int | None = None):
        super().__init__()
        self.main = nn.Sequential(
            self._pool(low_size),
            conv2d(low_ch, high_ch, self.POOLED, 1, 0, bias=False),
            Swish(),
            conv2d(high_ch, high_ch, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    @classmethod
    def _pool(cls, low_size: int | None) -> nn.Module:
        """Static pooling when the input size is known, adaptive otherwise.

        `adaptive_avg_pool2d` cannot be exported to ONNX once any input axis is
        dynamic -- the exporter needs a static input size to derive kernel and
        stride. Generator sizes are fixed by the architecture (only batch
        varies), so passing `low_size` there swaps in an equivalent fixed
        AvgPool2d and keeps the graph exportable. The discriminator is never
        exported and can keep the adaptive fallback.
        """
        if low_size is None:
            return nn.AdaptiveAvgPool2d(cls.POOLED)
        if low_size == cls.POOLED:
            return nn.Identity()
        if low_size % cls.POOLED != 0:
            raise ValueError(f"low_size {low_size} must be a multiple of {cls.POOLED}")
        factor = low_size // cls.POOLED
        return nn.AvgPool2d(factor, factor)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        return high * self.main(low)


class DownBlock(nn.Module):
    """Plain strided downsample."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.main = nn.Sequential(
            conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class DownBlockComp(nn.Module):
    """Residual downsample: strided path averaged with a pooled 1x1 path."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.main = nn.Sequential(
            conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.main(x) + self.direct(x)) / 2


class SimpleDecoder(nn.Module):
    """Tiny decoder from a discriminator feature map back to a 128x128 image.

    Used only for the self-supervised reconstruction task on real images, which
    is the regulariser that stops D memorising a small dataset. Deliberately
    weak (fixed 32-channel base) -- it must not be able to reconstruct without
    D's features actually carrying structure.
    """

    BASE = 32

    def __init__(self, in_ch: int, out_ch: int = 3):
        super().__init__()
        mult = {16: 4, 32: 2, 64: 2, 128: 1}
        nfc = {k: int(v * self.BASE) for k, v in mult.items()}
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            UpBlock(in_ch, nfc[16]),
            UpBlock(nfc[16], nfc[32]),
            UpBlock(nfc[32], nfc[64]),
            UpBlock(nfc[64], nfc[128]),
            conv2d(nfc[128], out_ch, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


def crop_quadrant(x: torch.Tensor, part: int) -> torch.Tensor:
    """One of the four quadrants of a feature map or image.

    D reconstructs a random quadrant as well as the whole image, which forces
    its features to stay locally informative rather than collapsing to a single
    global "is this real" statistic.
    """
    h, w = x.size(2) // 2, x.size(3) // 2
    if part == 0:
        return x[:, :, :h, :w]
    if part == 1:
        return x[:, :, :h, w:]
    if part == 2:
        return x[:, :, h:, :w]
    if part == 3:
        return x[:, :, h:, w:]
    raise ValueError(f"part must be 0-3, got {part}")


def resize(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)

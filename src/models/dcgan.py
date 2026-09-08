"""DCGAN at 64x64 (Radford et al., 2016) -- the Phase 1 baseline.

This exists to be beaten. Its jobs are to validate the data/training/sampling
loop end to end, to produce the "colourful blobs" reference that makes FastGAN's
improvement legible in docs/results.md, and to demonstrate mode collapse
first-hand on ~1.3k structurally distinct images so the Phase 2 mitigations
(DiffAugment, the self-supervised discriminator) have something to be measured
against. Do not expect quality from it.
"""

from __future__ import annotations

import torch
from torch import nn


class Generator(nn.Module):
    """z -> 4x4 -> 8 -> 16 -> 32 -> 64, transposed convs throughout.

    Kept deliberately faithful to the 2016 paper, checkerboard artefacts and
    all -- FastGAN's nearest-upsample-then-conv is one of the things being
    compared against.
    """

    def __init__(self, z_dim: int = 128, ngf: int = 512, nc: int = 3):
        super().__init__()
        self.z_dim = z_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf // 2, ngf // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf // 4, ngf // 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf // 8, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z.view(z.size(0), self.z_dim, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, ndf: int = 64, nc: int = 3):
        super().__init__()
        self.main = nn.Sequential(
            # No normalisation on the first layer, per the paper: it would wash
            # out the input distribution the discriminator needs to see.
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(-1)


@torch.no_grad()
def init_weights(module: nn.Module) -> None:
    name = module.__class__.__name__
    if "Conv" in name:
        module.weight.normal_(0.0, 0.02)
    elif "BatchNorm" in name:
        module.weight.normal_(1.0, 0.02)
        module.bias.fill_(0.0)


def build_models(z_dim: int = 128, ngf: int = 512, ndf: int = 64) -> tuple[Generator, Discriminator]:
    netG, netD = Generator(z_dim, ngf), Discriminator(ndf)
    netG.apply(init_weights)
    netD.apply(init_weights)
    return netG, netD

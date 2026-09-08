"""FastGAN generator and discriminator (Liu et al., ICLR 2021).

Two design choices carry the whole model on a ~1.3k-image dataset:

  Skip-Layer Excitation  Long-range channel gating in G (4x4 -> 64x64,
                         8x8 -> 128x128, 16x16 -> 256x256), which buys global
                         coherence without a deep, VRAM-hungry trunk.
  Self-supervised D      D reconstructs real images (whole, downsampled, and a
                         random quadrant) from its own features. On a dataset
                         this small D would otherwise memorise within a few
                         thousand steps; forcing it to stay a useful encoder is
                         what keeps the adversarial signal alive.

Both networks are dual-scale: G emits the full-resolution image *and* a 128px
version, and D scores both. The small branch is a cheap way to keep gradients
flowing to global structure rather than only to texture.
"""

from __future__ import annotations

import random

import torch
from torch import nn

from src.models.blocks import (
    DownBlock,
    DownBlockComp,
    GLU,
    InitLayer,
    SimpleDecoder,
    SkipLayerExcitation,
    UpBlock,
    UpBlockComp,
    conv2d,
    crop_quadrant,
    resize,
)

# Channel width per feature resolution, as a multiple of ngf/ndf.
G_MULT = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
D_MULT = {4: 16, 8: 16, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}

SMALL_SIZE = 128  # resolution of the auxiliary branch and every reconstruction


def _channels(mult: dict[int, float], base: int) -> dict[int, int]:
    return {k: max(int(v * base), 1) for k, v in mult.items()}


class Generator(nn.Module):
    def __init__(self, z_dim: int = 256, ngf: int = 64, im_size: int = 256, nc: int = 3):
        super().__init__()
        if im_size not in (256, 512, 1024):
            raise ValueError(f"im_size must be 256, 512 or 1024; got {im_size}")
        self.z_dim = z_dim
        self.im_size = im_size
        nfc = _channels(G_MULT, ngf)

        self.init = InitLayer(z_dim, nfc[4])
        self.feat_8 = UpBlockComp(nfc[4], nfc[8])
        self.feat_16 = UpBlock(nfc[8], nfc[16])
        self.feat_32 = UpBlockComp(nfc[16], nfc[32])
        self.feat_64 = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlockComp(nfc[64], nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])

        # low_size is passed explicitly so the pooling is static and the graph
        # stays ONNX-exportable with a dynamic batch axis.
        self.se_64 = SkipLayerExcitation(nfc[4], nfc[64], low_size=4)
        self.se_128 = SkipLayerExcitation(nfc[8], nfc[128], low_size=8)
        self.se_256 = SkipLayerExcitation(nfc[16], nfc[256], low_size=16)

        self.feat_512 = UpBlockComp(nfc[256], nfc[512]) if im_size >= 512 else None
        self.se_512 = (
            SkipLayerExcitation(nfc[32], nfc[512], low_size=32) if im_size >= 512 else None
        )
        self.feat_1024 = UpBlock(nfc[512], nfc[1024]) if im_size >= 1024 else None

        self.to_small = conv2d(nfc[SMALL_SIZE], nc, 1, 1, 0, bias=False)
        self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, z: torch.Tensor) -> list[torch.Tensor]:
        """Returns [full-resolution image, 128px image], both in [-1, 1]."""
        f4 = self.init(z)
        f8 = self.feat_8(f4)
        f16 = self.feat_16(f8)
        f32 = self.feat_32(f16)
        f64 = self.se_64(f4, self.feat_64(f32))
        f128 = self.se_128(f8, self.feat_128(f64))
        big = self.se_256(f16, self.feat_256(f128))

        if self.feat_512 is not None:
            big = self.se_512(f32, self.feat_512(big))
        if self.feat_1024 is not None:
            big = self.feat_1024(big)

        return [self.tanh(self.to_big(big)), self.tanh(self.to_small(f128))]


class Discriminator(nn.Module):
    def __init__(self, ndf: int = 64, im_size: int = 256, nc: int = 3):
        super().__init__()
        if im_size not in (256, 512, 1024):
            raise ValueError(f"im_size must be 256, 512 or 1024; got {im_size}")
        self.im_size = im_size
        nfc = _channels(D_MULT, ndf)

        # Bring any input resolution down to 256 before the shared trunk.
        if im_size == 1024:
            self.from_big = nn.Sequential(
                conv2d(nc, nfc[1024], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(nfc[1024], nfc[512], 4, 2, 1, bias=False),
                nn.BatchNorm2d(nfc[512]),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif im_size == 512:
            self.from_big = nn.Sequential(
                conv2d(nc, nfc[512], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.from_big = nn.Sequential(
                conv2d(nc, nfc[512], 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )

        # Spatial trace at im_size=256: 256 -> 128 -> 64 -> 32 -> 16 -> 8.
        # Block names track channel width (nfc keys), not spatial size.
        self.down_4 = DownBlockComp(nfc[512], nfc[256])
        self.down_8 = DownBlockComp(nfc[256], nfc[128])
        self.down_16 = DownBlockComp(nfc[128], nfc[64])
        self.down_32 = DownBlockComp(nfc[64], nfc[32])
        self.down_64 = DownBlockComp(nfc[32], nfc[16])

        self.se_16 = SkipLayerExcitation(nfc[512], nfc[64])
        self.se_32 = SkipLayerExcitation(nfc[256], nfc[32])
        self.se_64 = SkipLayerExcitation(nfc[128], nfc[16])

        # Patch-wise real/fake logits over the final 8x8 map (5x5 per sample).
        self.rf_big = nn.Sequential(
            conv2d(nfc[16], nfc[8], 1, 1, 0, bias=False),
            nn.BatchNorm2d(nfc[8]),
            nn.LeakyReLU(0.2, inplace=True),
            conv2d(nfc[8], 1, 4, 1, 0, bias=False),
        )

        self.from_small = nn.Sequential(
            conv2d(nc, nfc[256], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(nfc[256], nfc[128]),
            DownBlock(nfc[128], nfc[64]),
            DownBlock(nfc[64], nfc[32]),
        )
        self.rf_small = conv2d(nfc[32], 1, 4, 1, 0, bias=False)

        self.decoder_big = SimpleDecoder(nfc[16], nc)
        self.decoder_part = SimpleDecoder(nfc[32], nc)
        self.decoder_small = SimpleDecoder(nfc[32], nc)

    def forward(
        self, imgs: list[torch.Tensor] | torch.Tensor, real: bool = False, part: int | None = None
    ):
        """Score images; on reals also return the three reconstructions.

        `imgs` is either G's [big, small] pair or a single tensor, which is
        resized into that pair so real and fake take an identical path.
        """
        if not isinstance(imgs, (list, tuple)):
            imgs = [resize(imgs, self.im_size), resize(imgs, SMALL_SIZE)]

        f2 = self.from_big(imgs[0])
        f4 = self.down_4(f2)
        f8 = self.down_8(f4)
        f16 = self.se_16(f2, self.down_16(f8))
        f32 = self.se_32(f4, self.down_32(f16))
        f_last = self.se_64(f8, self.down_64(f32))

        logits_big = self.rf_big(f_last).view(-1)
        f_small = self.from_small(imgs[1])
        logits_small = self.rf_small(f_small).view(-1)
        logits = torch.cat([logits_big, logits_small])

        if not real:
            return logits

        if part is None:
            part = random.randint(0, 3)
        recons = [
            self.decoder_big(f_last),
            self.decoder_small(f_small),
            self.decoder_part(crop_quadrant(f32, part)),
        ]
        return logits, recons, part


@torch.no_grad()
def init_weights(module: nn.Module) -> None:
    """DCGAN-style init.

    Spectral norm reparameterises `weight` into `weight_orig` plus buffers, so
    initialising `weight` directly would be silently discarded on the next
    forward pass. Always target `weight_orig` when it is present.
    """
    name = module.__class__.__name__
    if "Conv" in name:
        target = getattr(module, "weight_orig", getattr(module, "weight", None))
        if target is not None:
            target.normal_(0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            module.bias.fill_(0.0)
    elif "BatchNorm" in name:
        if getattr(module, "weight", None) is not None:
            module.weight.normal_(1.0, 0.02)
        if getattr(module, "bias", None) is not None:
            module.bias.fill_(0.0)


def build_models(
    z_dim: int = 256, ngf: int = 64, ndf: int = 64, im_size: int = 256
) -> tuple[Generator, Discriminator]:
    netG = Generator(z_dim=z_dim, ngf=ngf, im_size=im_size)
    netD = Discriminator(ndf=ndf, im_size=im_size)
    netG.apply(init_weights)
    netD.apply(init_weights)
    return netG, netD

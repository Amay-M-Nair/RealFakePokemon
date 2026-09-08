"""DiffAugment (Zhao et al., NeurIPS 2020).

Differentiable augmentation applied to BOTH reals and fakes in BOTH the G and D
passes. That symmetry is the whole point: because the ops are differentiable,
gradients flow back through them to G, so G never learns to reproduce the
augmentation artefacts. Applying it only to reals -- the classic bug -- just
hands D a free tell and silently degrades results.

With ~1.3k structurally distinct images this is not optional. Without it D
memorises the training set within a few thousand steps and the adversarial
signal collapses.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rand_brightness(x: torch.Tensor) -> torch.Tensor:
    shift = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5
    return x + shift


def rand_saturation(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    factor = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2
    return (x - mean) * factor + mean


def rand_contrast(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    factor = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5
    return (x - mean) * factor + mean


def rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    """Random integer shift with zero padding, done by gather so it stays differentiable."""
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    tx = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    ty = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_b, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + tx + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + ty + 1, 0, x.size(3) + 1)
    padded = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    out = padded.permute(0, 2, 3, 1).contiguous()[grid_b, grid_x, grid_y]
    return out.permute(0, 3, 1, 2).contiguous()


def rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    """Zero a random square covering `ratio` of each side."""
    size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    off_x = torch.randint(0, x.size(2) + (1 - size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    off_y = torch.randint(0, x.size(3) + (1 - size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_b, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(size[0], dtype=torch.long, device=x.device),
        torch.arange(size[1], dtype=torch.long, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + off_x - size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + off_y - size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_b, grid_x, grid_y] = 0
    return x * mask.unsqueeze(1)


AUGMENT_FNS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}


def diff_augment(x, policy=("color", "translation", "cutout")):
    """Apply the policy. Accepts a tensor or FastGAN's [big, small] list.

    Each scale draws its own randomness, matching the reference implementation:
    the two scales are scored by separate discriminator branches, so they do not
    need to agree.
    """
    if isinstance(x, (list, tuple)):
        return [diff_augment(item, policy) for item in x]
    if not policy:
        return x
    for name in policy:
        if name not in AUGMENT_FNS:
            raise ValueError(f"unknown DiffAugment policy {name!r}")
        for fn in AUGMENT_FNS[name]:
            x = fn(x)
    return x.contiguous()

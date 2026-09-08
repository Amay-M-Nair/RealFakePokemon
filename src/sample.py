"""Sampling utilities: grids, truncation, interpolation, training GIFs.

Everything here samples from the EMA generator and takes an explicit seed, so a
given (checkpoint, seed, truncation) always yields the same image. The Django
app leans on that determinism to make every generated image a cacheable
permalink.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.config import get_device
from src.models.blocks import set_deterministic_noise
from src.models.fastgan import Generator


def to_pil(x: torch.Tensor) -> list[Image.Image]:
    """NCHW in [-1,1] -> list of PIL images."""
    arr = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    arr = arr.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(a) for a in arr]


def make_grid(images: list[Image.Image], cols: int | None = None, pad: int = 4) -> Image.Image:
    cols = cols or int(np.ceil(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / cols))
    w, h = images[0].size
    sheet = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), (255, 255, 255))
    for i, im in enumerate(images):
        c, r = i % cols, i // cols
        sheet.paste(im, (pad + c * (w + pad), pad + r * (h + pad)))
    return sheet


@torch.no_grad()
def generate(
    netG: Generator,
    z: torch.Tensor,
    truncation: float = 1.0,
    z_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full-resolution samples for the given latents.

    Truncation shrinks each latent toward the distribution mean: psi=1 is the
    untruncated model, lower values trade diversity for fidelity. `z_mean` is
    the empirical mean over many samples -- for a standard normal prior it is
    ~0, but using the measured value keeps this correct if the prior changes.
    """
    netG.eval()
    if truncation != 1.0:
        mean = torch.zeros_like(z[:1]) if z_mean is None else z_mean.to(z.device).view(1, -1)
        z = mean + truncation * (z - mean)
    return netG(z)[0]


def latents(count: int, z_dim: int, seed: int, device) -> torch.Tensor:
    """Seed-deterministic latents, reproducible across machines and devices.

    Generated on CPU with an explicit Generator: CUDA's RNG does not guarantee
    the same stream across driver or hardware versions, which would break the
    permalink promise the web app makes.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(count, z_dim, generator=g).to(device)


@torch.no_grad()
def save_grid(netG: Generator, z: torch.Tensor, path: Path, truncation: float = 1.0) -> torch.Tensor:
    """Write a sample grid and return it as CHW float for TensorBoard."""
    was_training = netG.training
    imgs = to_pil(generate(netG, z, truncation))
    grid = make_grid(imgs)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)
    if was_training:
        netG.train()
    return torch.from_numpy(np.asarray(grid)).permute(2, 0, 1).float() / 255.0


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical interpolation.

    Linear interpolation between Gaussian latents passes through a
    low-magnitude region the generator never saw in training, which shows up as
    washed-out frames mid-transition. Slerp keeps the norm roughly constant.
    """
    a_n, b_n = a / a.norm(dim=-1, keepdim=True), b / b.norm(dim=-1, keepdim=True)
    omega = torch.acos((a_n * b_n).sum(-1, keepdim=True).clamp(-1, 1))
    sin_omega = torch.sin(omega)
    if (sin_omega.abs() < 1e-6).all():
        return (1 - t) * a + t * b
    return (torch.sin((1 - t) * omega) / sin_omega) * a + (torch.sin(t * omega) / sin_omega) * b


@torch.no_grad()
def interpolate(
    netG: Generator, seed_a: int, seed_b: int, steps: int = 8, truncation: float = 1.0
) -> list[Image.Image]:
    device = next(netG.parameters()).device
    za = latents(1, netG.z_dim, seed_a, device)
    zb = latents(1, netG.z_dim, seed_b, device)
    zs = torch.cat([slerp(za, zb, t) for t in np.linspace(0, 1, steps)])
    return to_pil(generate(netG, zs, truncation))


def load_generator(checkpoint: Path, device=None, use_ema: bool = True) -> Generator:
    """Rebuild the generator from a checkpoint, defaulting to the EMA weights."""
    device = device or get_device()
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    netG = Generator(z_dim=cfg["z_dim"], ngf=cfg["ngf"], im_size=cfg["resolution"]).to(device)
    netG.load_state_dict(ckpt["ema"] if use_ema else ckpt["G"])
    netG.eval()
    # Everything downstream of a checkpoint -- sampling, metrics, export -- must
    # be reproducible from the seed alone, so the stochastic noise term is off.
    set_deterministic_noise(netG)
    return netG


def training_gif(sample_dir: Path, out: Path, duration: int = 120) -> None:
    """Assemble the periodic fixed-seed grids into a progress animation."""
    frames = sorted(sample_dir.glob("*.png"))
    if not frames:
        raise SystemExit(f"no sample grids in {sample_dir}")
    images = [Image.open(f).convert("RGB") for f in frames]
    images[0].save(out, save_all=True, append_images=images[1:], duration=duration, loop=0)
    print(f"{len(images)} frames -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path("samples/grid.png"))
    parser.add_argument("--count", type=int, default=36)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--truncation", type=float, default=1.0)
    parser.add_argument("--interpolate", nargs=2, type=int, metavar=("SEED_A", "SEED_B"))
    parser.add_argument("--raw", action="store_true", help="use live G weights, not the EMA copy")
    args = parser.parse_args()

    device = get_device()
    netG = load_generator(args.checkpoint, device, use_ema=not args.raw)

    if args.interpolate:
        imgs = interpolate(netG, *args.interpolate, steps=args.count, truncation=args.truncation)
        grid = make_grid(imgs, cols=len(imgs))
    else:
        z = latents(args.count, netG.z_dim, args.seed, device)
        grid = make_grid(to_pil(generate(netG, z, args.truncation)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

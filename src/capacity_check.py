"""Generator capacity check -- can G represent real images at all?

    python -m src.capacity_check --images 16 --steps 1500

Fits the generator to a fixed batch of real images with a plain reconstruction
objective, learning one latent per image alongside the weights. No
discriminator, no adversarial dynamics.

Why this and not the usual "overfit 20 images with augmentation off": that test
is not diagnostic for a GAN. With a handful of images and no augmentation the
discriminator memorises them within a couple of thousand steps, its accuracy
saturates near 1.0, and the generator stops receiving useful gradient. The run
then fails whether or not the architecture is sound, so it cannot tell those two
cases apart -- which is exactly what happened here (see docs/results.md).

Removing D entirely isolates the question actually worth answering before
committing tens of hours: does the generator have the capacity and the gradient
path to produce these images?
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from src.config import DataConfig, FastGANConfig, SAMPLES_DIR, get_device
from src.data.dataset import PokemonArtwork
from src.models.blocks import set_deterministic_noise
from src.models.fastgan import Generator
from src.sample import make_grid, to_pil


def run(args) -> float:
    cfg = FastGANConfig()
    device = get_device()
    torch.manual_seed(0)

    dataset = PokemonArtwork(resolution=cfg.resolution, mirror=False)
    target = torch.stack([dataset[i] for i in range(args.images)]).to(device)

    netG = Generator(z_dim=cfg.z_dim, ngf=cfg.ngf, im_size=cfg.resolution).to(device)
    set_deterministic_noise(netG)  # keep the objective a deterministic function of z
    netG.train()

    # One free latent per target, optimised jointly with the weights.
    z = torch.randn(args.images, cfg.z_dim, device=device, requires_grad=True)
    opt = torch.optim.Adam([{"params": netG.parameters(), "lr": 2e-3},
                            {"params": [z], "lr": 1e-2}], betas=(0.5, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print(f"fitting {args.images} images for {args.steps} steps (no discriminator)")
    first = last = None
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
            out = netG(z)[0]
            loss = F.l1_loss(out, target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if step == 0:
            first = loss.item()
        if (step + 1) % max(1, args.steps // 10) == 0:
            print(f"  step {step+1:>5}/{args.steps}  L1 {loss.item():.4f}", flush=True)
        last = loss.item()

    out_dir = SAMPLES_DIR / "capacity_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        netG.eval()
        recon = netG(z)[0]
    cols = min(args.images, 8)
    make_grid(to_pil(recon.float().cpu()), cols=cols).save(out_dir / "reconstruction.png")
    make_grid(to_pil(target.cpu()), cols=cols).save(out_dir / "target.png")

    print(f"\nL1: {first:.4f} -> {last:.4f}  ({100*(1-last/first):.0f}% reduction)")
    print(f"wrote {out_dir/'target.png'} and {out_dir/'reconstruction.png'}")
    verdict = "PASS" if last < args.threshold else "FAIL"
    print(f"VERDICT: {verdict} (threshold L1 < {args.threshold})")
    if verdict == "FAIL":
        print("The generator cannot represent real images -- fix the architecture "
              "before spending hours on adversarial training.")
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--threshold", type=float, default=0.08)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

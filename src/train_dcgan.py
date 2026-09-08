"""DCGAN baseline training (Phase 1).

    python -m src.train_dcgan --steps 30000

Kept as a separate script rather than a branch inside src/train.py: DCGAN has no
reconstruction term, no dual-scale output and no EMA in the original, so
threading it through the FastGAN loop would add flags to every line for a model
whose whole purpose is to be simple and to lose.

What this is for: validating the loop end to end, producing the "colourful
blobs" reference that makes FastGAN's improvement legible in docs/results.md,
and demonstrating mode collapse first-hand on ~1.3k distinct shapes.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.config import CHECKPOINT_DIR, RUNS_DIR, SAMPLES_DIR, DataConfig, DCGANConfig, get_device
from src.data.dataset import build_loader, infinite
from src.models.dcgan import build_models
from src.sample import make_grid, to_pil


def save_samples(netG, z, path: Path) -> torch.Tensor:
    was_training = netG.training
    netG.eval()
    with torch.no_grad():
        grid = make_grid(to_pil(netG(z)))
    if was_training:
        netG.train()
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)
    import numpy as np

    return torch.from_numpy(np.asarray(grid)).permute(2, 0, 1).float() / 255.0


def train(args) -> None:
    cfg = DCGANConfig()
    if args.steps:
        cfg.total_steps = args.steps
    if args.batch_size:
        cfg.batch_size = args.batch_size

    device = get_device()
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    data_cfg = DataConfig(resolution=cfg.resolution)
    loader = build_loader(data_cfg, batch_size=cfg.batch_size, num_workers=args.workers)
    print(f"dataset: {len(loader.dataset)} images at {cfg.resolution}px")
    batches = infinite(loader)

    netG, netD = build_models(cfg.z_dim, cfg.g_channels, cfg.d_channels)
    netG, netD = netG.to(device), netD.to(device)
    optG = torch.optim.Adam(netG.parameters(), lr=cfg.lr, betas=cfg.betas)
    optD = torch.optim.Adam(netD.parameters(), lr=cfg.lr, betas=cfg.betas)

    run_dir, ckpt_dir, sample_dir = RUNS_DIR / args.name, CHECKPOINT_DIR / args.name, SAMPLES_DIR / args.name
    writer = SummaryWriter(run_dir)
    (run_dir / "config.json").parent.mkdir(parents=True, exist_ok=True)

    start_step = 0
    fixed_z = torch.randn(36, cfg.z_dim, device=device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        netG.load_state_dict(ckpt["G"]); netD.load_state_dict(ckpt["D"])
        optG.load_state_dict(ckpt["optG"]); optD.load_state_dict(ckpt["optD"])
        fixed_z = ckpt["fixed_z"].to(device)
        start_step = ckpt["step"]
        print(f"resumed from {args.resume} at step {start_step}")

    print(f"training {start_step} -> {cfg.total_steps} (batch {cfg.batch_size})")
    t0 = time.time()
    running = {"d": 0.0, "g": 0.0, "acc_real": 0.0, "acc_fake": 0.0}

    for step in range(start_step, cfg.total_steps):
        real = next(batches).to(device, non_blocking=True)
        batch = real.size(0)

        # ---- discriminator: BCE on reals and detached fakes ----
        optD.zero_grad(set_to_none=True)
        logits_real = netD(real)
        loss_real = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))

        fake = netG(torch.randn(batch, cfg.z_dim, device=device))
        logits_fake = netD(fake.detach())
        loss_fake = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))

        loss_d = loss_real + loss_fake
        loss_d.backward()
        optD.step()

        # ---- generator: non-saturating loss (maximise log D(G(z))) ----
        optG.zero_grad(set_to_none=True)
        logits = netD(fake)
        loss_g = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        loss_g.backward()
        optG.step()

        running["d"] += loss_d.item()
        running["g"] += loss_g.item()
        running["acc_real"] += (logits_real > 0).float().mean().item()
        running["acc_fake"] += (logits_fake < 0).float().mean().item()

        if (step + 1) % 100 == 0:
            entry = {k: v / 100 for k, v in running.items()}
            for k, v in entry.items():
                writer.add_scalar(f"train/{k}", v, step + 1)
            rate = (step + 1 - start_step) / max(time.time() - t0, 1e-9)
            print(
                f"step {step+1:>6}/{cfg.total_steps}  d {entry['d']:.3f}  g {entry['g']:.3f}  "
                f"acc(r/f) {entry['acc_real']:.2f}/{entry['acc_fake']:.2f}  {rate:.1f} it/s",
                flush=True,
            )
            running = dict.fromkeys(running, 0.0)

        if (step + 1) % 1000 == 0:
            writer.add_image("samples", save_samples(netG, fixed_z, sample_dir / f"{step+1:07d}.png"), step + 1)

        if (step + 1) % 2000 == 0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            tmp = ckpt_dir / "latest.tmp"
            torch.save(
                {"config": asdict(cfg), "step": step + 1, "G": netG.state_dict(),
                 "D": netD.state_dict(), "optG": optG.state_dict(), "optD": optD.state_dict(),
                 "fixed_z": fixed_z.cpu()},
                tmp,
            )
            tmp.replace(ckpt_dir / "latest.pt")

    writer.close()
    print(f"done in {(time.time()-t0)/60:.1f} min")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="dcgan64")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", type=Path, default=None)
    train(parser.parse_args())


if __name__ == "__main__":
    main()

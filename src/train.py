"""FastGAN training loop.

    python -m src.train --steps 100000
    python -m src.train --resume checkpoints/fastgan/latest.pt

Resume is first-class, not an afterthought: a full run is ~37h on a 4GB laptop
GPU and has to survive being stopped every night, and Kaggle's 12h session cap
makes it mandatory for the cloud runs too. Every checkpoint carries G, D, the
EMA copy, both optimizers, the AMP scaler and the fixed sample latents, so a
resumed run continues rather than restarts.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from src.augment import diff_augment
from src.config import CHECKPOINT_DIR, RUNS_DIR, SAMPLES_DIR, DataConfig, FastGANConfig, get_device
from src.data.dataset import build_loader, infinite
from src.losses import ReconstructionLoss, d_hinge_loss, g_loss
from src.models.blocks import crop_quadrant
from src.models.ema import ModelEMA
from src.models.fastgan import build_models
from src.sample import save_grid


def _sync_state(cfg: FastGANConfig, data_cfg: DataConfig, args) -> None:
    """CLI overrides win over dataclass defaults."""
    if args.steps is not None:
        cfg.total_steps = args.steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.recon_loss is not None:
        cfg.recon_loss = args.recon_loss
    if args.resolution is not None:
        cfg.resolution = data_cfg.resolution = args.resolution
    if args.lr is not None:
        cfg.lr = args.lr


def save_checkpoint(path: Path, *, cfg, step, G, D, ema, optG, optD, scaler, fixed_z, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "config": asdict(cfg),
            "step": step,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "ema": ema.state_dict(),
            "optG": optG.state_dict(),
            "optD": optD.state_dict(),
            "scaler": scaler.state_dict(),
            "fixed_z": fixed_z.cpu(),
            "history": history,
        },
        tmp,
    )
    tmp.replace(path)  # atomic: a crash mid-save never corrupts the last good checkpoint


def train(args) -> None:
    cfg, data_cfg = FastGANConfig(), DataConfig()
    _sync_state(cfg, data_cfg, args)
    device = get_device()
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    run_dir = RUNS_DIR / args.name
    ckpt_dir = CHECKPOINT_DIR / args.name
    sample_dir = SAMPLES_DIR / args.name
    for d in (run_dir, ckpt_dir, sample_dir):
        d.mkdir(parents=True, exist_ok=True)

    if args.no_augment:
        cfg.diffaug_policy = ()
    loader = build_loader(
        data_cfg, batch_size=cfg.batch_size, num_workers=args.workers, limit=args.overfit
    )
    print(
        f"dataset: {len(loader.dataset)} images at {cfg.resolution}px"
        f"{' (OVERFIT SANITY CHECK)' if args.overfit else ''}, "
        f"augment={cfg.diffaug_policy or 'off'}"
    )
    batches = infinite(loader)

    G, D = build_models(cfg.z_dim, cfg.ngf, cfg.ndf, cfg.resolution)
    G, D = G.to(device), D.to(device)
    if cfg.channels_last:
        G = G.to(memory_format=torch.channels_last)
        D = D.to(memory_format=torch.channels_last)

    recon = ReconstructionLoss(cfg.recon_loss).to(device)
    ema = ModelEMA(G, cfg.ema_decay, cfg.ema_warmup_steps)
    optG = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=cfg.betas)
    optD = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=cfg.betas)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    start_step = 0
    history: list[dict] = []
    fixed_z = torch.randn(cfg.fixed_sample_count, cfg.z_dim, device=device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)

        # Architecture must match or the state dicts silently belong to a
        # different model. Checked explicitly because a resume that quietly
        # restarts from scratch would waste a night before anyone noticed.
        saved = ckpt.get("config", {})
        mismatched = {
            key: (saved[key], getattr(cfg, key))
            for key in ("resolution", "z_dim", "ngf", "ndf")
            if key in saved and saved[key] != getattr(cfg, key)
        }
        if mismatched:
            details = ", ".join(f"{k}: checkpoint={a} current={b}" for k, (a, b) in mismatched.items())
            raise SystemExit(f"cannot resume -- architecture differs ({details})")

        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        ema.load_state_dict(ckpt["ema"])
        optG.load_state_dict(ckpt["optG"])
        optD.load_state_dict(ckpt["optD"])
        scaler.load_state_dict(ckpt["scaler"])
        fixed_z = ckpt["fixed_z"].to(device)
        start_step = ckpt["step"]
        history = ckpt.get("history", [])
        print(f"resumed from {args.resume} at step {start_step}")

    writer = SummaryWriter(run_dir)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    autocast = torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled())
    mem_fmt = torch.channels_last if cfg.channels_last else torch.contiguous_format

    print(f"training {start_step} -> {cfg.total_steps} (batch {cfg.batch_size}, recon {cfg.recon_loss})")
    t0 = t_window = time.time()
    running = {"d": 0.0, "g": 0.0, "recon": 0.0, "acc_real": 0.0, "acc_fake": 0.0}

    for step in range(start_step, cfg.total_steps):
        # ---- discriminator -------------------------------------------------
        # Fakes are generated under no_grad here: D's update does not need G's
        # graph, and skipping it is a large chunk of the memory saving that lets
        # batch 8 fit in 4GB.
        optD.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum):
            real = next(batches).to(device, non_blocking=True).to(memory_format=mem_fmt)
            with autocast:
                with torch.no_grad():
                    fake = G(torch.randn(cfg.batch_size, cfg.z_dim, device=device))
                real_aug = diff_augment(real, cfg.diffaug_policy)
                fake_aug = diff_augment([f.detach() for f in fake], cfg.diffaug_policy)

                logits_r, recons, part = D(real_aug, real=True)
                logits_f = D(fake_aug, real=False)

                adv = d_hinge_loss(logits_r, True) + d_hinge_loss(logits_f, False)
                rec = (
                    recon(recons[0], real_aug)
                    + recon(recons[1], real_aug)
                    + recon(recons[2], crop_quadrant(real_aug, part))
                )
                loss_d = (adv + cfg.recon_weight * rec) / cfg.grad_accum
            scaler.scale(loss_d).backward()
        scaler.step(optD)

        # ---- generator -----------------------------------------------------
        # This backward also writes gradients into D; they are discarded by the
        # zero_grad at the top of the next iteration.
        optG.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum):
            with autocast:
                fake = G(torch.randn(cfg.batch_size, cfg.z_dim, device=device))
                loss_g = g_loss(D(diff_augment(fake, cfg.diffaug_policy))) / cfg.grad_accum
            scaler.scale(loss_g).backward()
        scaler.step(optG)
        scaler.update()

        ema.update(G, step)

        running["d"] += adv.item()
        running["g"] += loss_g.item() * cfg.grad_accum
        running["recon"] += rec.item()
        # Fraction of patches D gets right. Sustained ~1.0 on both means D has
        # memorised the set -- the failure mode DiffAugment exists to prevent.
        running["acc_real"] += (logits_r > 0).float().mean().item()
        running["acc_fake"] += (logits_f < 0).float().mean().item()

        if (step + 1) % cfg.log_every == 0:
            n = cfg.log_every
            # Windowed, not cumulative. A cumulative average never recovers from
            # a stall -- one overnight pause dragged this from 0.73 to 0.46 and
            # inflated the ETA from 37h to 56h while the loop was actually
            # running at 1.1 it/s. The window reports what is happening now.
            now = time.time()
            rate = n / max(now - t_window, 1e-9)
            t_window = now
            entry = {"step": step + 1, **{k: v / n for k, v in running.items()}}
            history.append(entry)
            for k, v in entry.items():
                if k != "step":
                    writer.add_scalar(f"train/{k}", v, step + 1)
            writer.add_scalar("train/steps_per_sec", rate, step + 1)
            eta_h = (cfg.total_steps - step - 1) / max(rate, 1e-9) / 3600
            print(
                f"step {step+1:>7}/{cfg.total_steps}  "
                f"d {entry['d']:.3f}  g {entry['g']:.3f}  recon {entry['recon']:.3f}  "
                f"acc(r/f) {entry['acc_real']:.2f}/{entry['acc_fake']:.2f}  "
                f"{rate:.2f} it/s  eta {eta_h:.1f}h",
                flush=True,
            )
            running = dict.fromkeys(running, 0.0)

        if (step + 1) % cfg.sample_every == 0:
            # Sampling is cosmetic; training is not. On a 4GB card the batch-16
            # grid is the memory high-water mark of the whole loop, and losing a
            # 37-hour run to an OOM in a preview image would be absurd -- so a
            # failure here is logged and skipped rather than fatal.
            try:
                grid = save_grid(ema.ema, fixed_z, sample_dir / f"{step+1:07d}.png")
                writer.add_image("samples/ema", grid, step + 1)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  [step {step+1}] OOM while sampling; skipped, training continues", flush=True)

        if (step + 1) % cfg.checkpoint_every == 0:
            names = ["latest.pt"]
            if args.keep_all:
                names.append(f"step_{step+1:07d}.pt")
            for name in names:
                save_checkpoint(
                    ckpt_dir / name,
                    cfg=cfg, step=step + 1, G=G, D=D, ema=ema,
                    optG=optG, optD=optD, scaler=scaler, fixed_z=fixed_z, history=history,
                )

    save_checkpoint(
        ckpt_dir / "final.pt",
        cfg=cfg, step=cfg.total_steps, G=G, D=D, ema=ema,
        optG=optG, optD=optD, scaler=scaler, fixed_z=fixed_z, history=history,
    )
    writer.close()
    print(f"done in {(time.time()-t0)/3600:.1f}h -> {ckpt_dir/'final.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="fastgan")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--recon-loss", choices=("mse", "vgg"), default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--keep-all", action="store_true", help="keep every periodic checkpoint, not just latest"
    )
    parser.add_argument(
        "--overfit",
        type=int,
        default=None,
        metavar="N",
        help="train on only N images -- the sanity check that the architecture "
        "can fit anything at all. Pair with --no-augment.",
    )
    parser.add_argument("--no-augment", action="store_true", help="disable DiffAugment")
    train(parser.parse_args())


if __name__ == "__main__":
    main()

"""Evaluation: KID, FID, and a nearest-neighbour memorisation check.

Read the caveats before quoting any number from here.

KID is the headline metric, not FID. FID's covariance estimate is badly biased
below roughly 2,048 samples and this dataset has ~2.2k images total, so a bare
FID on it is close to meaningless -- quoting one anyway is the standard mistake
in Pokemon-GAN write-ups. KID's unbiased MMD estimator stays usable at this
scale. FID is still reported, for comparability, with its sample count attached.

These use torchvision's Inception-V3 weights, which differ slightly from the
original TF-Inception graph that published FID numbers were computed against.
Numbers here are directly comparable BETWEEN runs in this repo, and only roughly
comparable to numbers in papers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torch import nn
from tqdm import tqdm

from src.config import DataConfig, get_device
from src.data.dataset import PokemonArtwork
from src.sample import generate, latents, load_generator, make_grid, to_pil

INCEPTION_SIZE = 299


class InceptionFeatures(nn.Module):
    """2048-d pool3 activations, the standard FID/KID feature space."""

    def __init__(self):
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3

        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        net.fc = nn.Identity()
        self.net = net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x is NCHW in [-1, 1]."""
        x = F.interpolate(
            (x + 1) / 2, size=(INCEPTION_SIZE, INCEPTION_SIZE), mode="bilinear", align_corners=False
        )
        return self.net((x - self.mean) / self.std)


def frechet_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    diff = mu_a - mu_b

    covmean, _ = linalg.sqrtm(cov_a.dot(cov_b), disp=False)
    if not np.isfinite(covmean).all():
        # Singular product; nudge the diagonal rather than returning a NaN.
        offset = np.eye(cov_a.shape[0]) * eps
        covmean = linalg.sqrtm((cov_a + offset).dot(cov_b + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(cov_a) + np.trace(cov_b) - 2 * np.trace(covmean))


def kernel_distance(
    a: np.ndarray, b: np.ndarray, subsets: int = 100, subset_size: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """Unbiased KID (polynomial-kernel MMD^2), averaged over random subsets.

    Returns (mean, std). Unlike FID this has no bias term that grows as the
    sample count shrinks, which is why it is the metric to trust here.
    """
    rng = np.random.default_rng(seed)
    n = min(subset_size, len(a), len(b))
    d = a.shape[1]
    scores = []
    for _ in range(subsets):
        x = a[rng.choice(len(a), n, replace=False)]
        y = b[rng.choice(len(b), n, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        # Exclude the diagonal for the within-set terms -- that is what makes
        # the estimator unbiased.
        m = n * (n - 1)
        scores.append(
            (kxx.sum() - np.trace(kxx)) / m + (kyy.sum() - np.trace(kyy)) / m - 2 * kxy.mean()
        )
    return float(np.mean(scores)), float(np.std(scores))


@torch.no_grad()
def features_from_dataset(model: InceptionFeatures, dataset, batch_size: int, device) -> np.ndarray:
    out = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="real features"):
        batch = torch.stack([dataset[j] for j in range(i, min(i + batch_size, len(dataset)))])
        out.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def features_from_generator(
    model: InceptionFeatures, netG, count: int, batch_size: int, device, truncation: float = 1.0
) -> np.ndarray:
    out = []
    for i in tqdm(range(0, count, batch_size), desc="fake features"):
        n = min(batch_size, count - i)
        z = latents(n, netG.z_dim, seed=10_000 + i, device=device)
        out.append(model(generate(netG, z, truncation)).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def nearest_neighbours(
    netG, dataset, device, count: int = 8, seed: int = 123, truncation: float = 1.0
) -> Image.Image:
    """Panel pairing generated samples with their closest training image.

    With ~1.3k structurally distinct images, showing that the model is not
    simply reproducing training art is a core result, not an optional extra.
    Rows alternate: generated on top, its nearest real neighbour below.
    """
    from src.losses import VGGPerceptualLoss

    vgg = VGGPerceptualLoss().to(device).eval()

    def embed(x: torch.Tensor) -> torch.Tensor:
        h = vgg._normalize(x)
        feats = []
        for i, layer in enumerate(vgg.features):
            h = layer(h)
            if i in vgg.LAYERS:
                feats.append(F.adaptive_avg_pool2d(h, 1).flatten(1))
        v = torch.cat(feats, dim=1)
        return v / v.norm(dim=1, keepdim=True)

    real_embeds, real_imgs = [], []
    for i in tqdm(range(0, len(dataset), 32), desc="embedding reals"):
        batch = torch.stack([dataset[j] for j in range(i, min(i + 32, len(dataset)))]).to(device)
        real_embeds.append(embed(batch).cpu())
        real_imgs.append(batch.cpu())
    real_embeds = torch.cat(real_embeds)
    real_imgs = torch.cat(real_imgs)

    z = latents(count, netG.z_dim, seed, device)
    fake = generate(netG, z, truncation)
    sims = embed(fake).cpu() @ real_embeds.T
    nn_idx = sims.argmax(dim=1)

    rows = to_pil(fake.cpu()) + to_pil(real_imgs[nn_idx])
    return make_grid(rows, cols=count)


def evaluate(args) -> dict:
    device = get_device()
    netG = load_generator(args.checkpoint, device)
    inception = InceptionFeatures().to(device)
    dataset = PokemonArtwork(
        resolution=args.resolution, mirror=False, include_shiny=not args.no_shiny
    )

    real = features_from_dataset(inception, dataset, args.batch_size, device)
    fake = features_from_generator(
        inception, netG, args.samples, args.batch_size, device, args.truncation
    )

    fid = frechet_distance(real, fake)
    kid_mean, kid_std = kernel_distance(
        real, fake, subsets=args.kid_subsets, subset_size=min(args.kid_subset_size, len(real))
    )
    results = {
        "checkpoint": str(args.checkpoint),
        "truncation": args.truncation,
        "n_real": len(real),
        "n_fake": len(fake),
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "fid": fid,
    }

    print("\n--- results ---")
    print(f"KID  {kid_mean:.5f} +/- {kid_std:.5f}   <- headline metric")
    print(f"FID  {fid:.2f}   (n_real={len(real)}; biased below ~2048 samples, treat as indicative)")

    if args.nn_panel:
        panel = nearest_neighbours(netG, dataset, device, count=args.nn_count)
        args.nn_panel.parent.mkdir(parents=True, exist_ok=True)
        panel.save(args.nn_panel)
        print(f"nearest-neighbour panel -> {args.nn_panel} (top row generated, bottom row nearest real)")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=DataConfig().resolution)
    parser.add_argument("--truncation", type=float, default=1.0)
    parser.add_argument("--no-shiny", action="store_true")
    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument("--kid-subset-size", type=int, default=1000)
    parser.add_argument("--nn-panel", type=Path, default=Path("docs/assets/nearest_neighbours.png"))
    parser.add_argument("--nn-count", type=int, default=8)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    results = evaluate(args)
    if args.json:
        import json

        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()

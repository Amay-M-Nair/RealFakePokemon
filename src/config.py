"""Paths and hyperparameters.

Defaults throughout target a single RTX 3050 Laptop (4GB VRAM). Anything that
would not fit in 4GB is called out in the field comment.
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch

# --- Paths -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARTWORK_DIR = RAW_DIR / "official-artwork"
PROCESSED_DIR = DATA_DIR / "processed"
META_PATH = RAW_DIR / "meta.json"
MANIFEST_PATH = RAW_DIR / "manifest.json"
CHECKPOINT_DIR = ROOT / "checkpoints"
RUNS_DIR = ROOT / "runs"
SAMPLES_DIR = ROOT / "samples"

# --- Dataset source ----------------------------------------------------------

SPRITES_REPO = "PokeAPI/sprites"
SPRITES_BRANCH = "master"
ARTWORK_PREFIX = "sprites/pokemon/other/official-artwork"
RAW_BASE_URL = f"https://raw.githubusercontent.com/{SPRITES_REPO}/{SPRITES_BRANCH}"

# Gen 9 starts at Sprigatito. Form variants use synthetic ids >= 10000, so a
# plain `dex_id >= GEN9_START` test also sweeps in Gen 1-8 forms; use the
# manifest's `generation` field for anything that must be exact.
GEN9_START = 906
NATIONAL_DEX_MAX = 1025


@dataclass
class DataConfig:
    """Preprocessing. `tight_crop` is the main quality lever worth ablating."""

    resolution: int = 256
    include_shiny: bool = True
    # Crop to the alpha bounding box before resizing, which normalises subject
    # scale across the set (raw artwork frames Wailord and Joltik very
    # differently). Set False to build the uncropped control set.
    tight_crop: bool = True
    # Fraction of the cropped square added as padding on every side.
    crop_margin: float = 0.06
    background: tuple[int, int, int] = (255, 255, 255)
    # Two images are duplicates only if they match on BOTH axes below.
    # pHash alone is computed on luminance and so cannot tell a shiny from its
    # base form, which would discard the palette diversity shinies are here for.
    dedupe_threshold: int = 4          # pHash Hamming distance
    color_threshold: float = 0.02      # mean abs RGB difference, 0-1 scale
    num_workers: int = 4


@dataclass
class DCGANConfig:
    """Phase 1 baseline. Exists to validate the loop, not to produce quality."""

    resolution: int = 64
    z_dim: int = 128
    g_channels: int = 512
    d_channels: int = 64
    batch_size: int = 64
    lr: float = 2e-4
    betas: tuple[float, float] = (0.5, 0.999)
    total_steps: int = 30_000


@dataclass
class FastGANConfig:
    """Phase 2 main model (Liu et al., ICLR 2021)."""

    resolution: int = 256
    z_dim: int = 256

    ngf: int = 64
    ndf: int = 64

    # Measured on an RTX 3050 Laptop (4GB): batch 8 with the VGG
    # reconstruction loss peaks at 2.72 GiB and runs ~1.4 s/step, i.e. ~37h per
    # 100k steps. Batch 8 is also the paper's own setting, so no accumulation
    # is needed -- raising grad_accum doubles wall clock for a marginal gain.
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 2e-4
    betas: tuple[float, float] = (0.5, 0.999)
    # ~37h on the 4GB card at batch 8; matches the figure quoted in the README.
    total_steps: int = 100_000

    # Discriminator self-supervised reconstruction, applied to real images only.
    # "vgg" is the paper-faithful perceptual term and costs only ~0.4 GiB more
    # than "mse" here, so it is the default; drop to "mse" if a run OOMs.
    recon_loss: str = "vgg"
    recon_weight: float = 1.0
    recon_size: int = 128

    ema_decay: float = 0.999
    ema_warmup_steps: int = 1_000

    # DiffAugment policies, applied to real and fake in both G and D passes.
    diffaug_policy: tuple[str, ...] = ("color", "translation", "cutout")

    amp: bool = True
    channels_last: bool = True

    log_every: int = 100
    sample_every: int = 1_000
    checkpoint_every: int = 2_000
    fixed_sample_count: int = 16
    seed: int = 0


@dataclass
class TrainState:
    """Everything a run needs to resume byte-identically across nights."""

    step: int = 0
    epoch: int = 0
    best_kid: float = float("inf")
    fixed_z: list = field(default_factory=list)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def processed_dir(resolution: int, tight_crop: bool = True) -> Path:
    """Cropped and uncropped sets live side by side so they can be compared."""
    suffix = "" if tight_crop else "-uncropped"
    return PROCESSED_DIR / f"{resolution}{suffix}"

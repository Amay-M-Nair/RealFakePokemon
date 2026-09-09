"""Torch Dataset over the preprocessed artwork.

Phase 1 deliverable: this exists to prove the manifest format is actually usable
by a training loop. Phase 2 builds the model that consumes it.

    ds = PokemonArtwork(resolution=256)                    # unconditional
    ds = PokemonArtwork(resolution=256, labelled_only=True) # (image, class_idx)
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.config import DataConfig, processed_dir


class PokemonArtwork(Dataset):
    """Preprocessed artwork as float tensors in [-1, 1], NCHW.

    Horizontal flip is the only augmentation applied here. Anything else belongs
    in the training step, where it can be applied identically to real and
    generated images -- doing it in the Dataset would only ever touch the reals
    and quietly bias the discriminator.
    """

    def __init__(
        self,
        resolution: int = 256,
        tight_crop: bool = True,
        mirror: bool = True,
        include_shiny: bool = True,
        labelled_only: bool = False,
        label_field: str = "top",
    ):
        self.root = processed_dir(resolution, tight_crop)
        manifest = self.root / "manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(
                f"{manifest} missing -- run "
                f"`python -m src.data.preprocess --resolution {resolution}` first"
            )

        with manifest.open(newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["kept"] == "1"]
        if not include_shiny:
            rows = [r for r in rows if r["shiny"] == "0"]
        if labelled_only:
            rows = [r for r in rows if r[label_field]]
        if not rows:
            raise RuntimeError(f"no usable rows in {manifest}")

        self.rows = rows
        self.resolution = resolution
        self.mirror = mirror
        self.label_field = label_field
        # Sorted so class indices are stable across runs and machines.
        self.classes = sorted({r[label_field] for r in rows if r[label_field]})
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {c: 0 for c in self.classes}
        for r in self.rows:
            if r[self.label_field]:
                counts[r[self.label_field]] += 1
        return counts

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        with Image.open(self.root / row["file"]) as im:
            img = im.convert("RGB")

        if self.mirror and torch.rand(1).item() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
        x = x.view(img.size[1], img.size[0], 3).permute(2, 0, 1).float()
        x = x.div_(127.5).sub_(1.0)  # [0,255] -> [-1,1], matching a tanh generator

        label = row[self.label_field]
        # -1 marks "unlabelled" so an unconditional run can ignore it and a
        # conditional run can assert it never appears.
        return x, self.class_to_idx.get(label, -1)


def build_loader(
    cfg: DataConfig | None = None,
    batch_size: int = 8,
    num_workers: int | None = None,
    **kwargs,
) -> DataLoader:
    cfg = cfg or DataConfig()
    workers = cfg.num_workers if num_workers is None else num_workers
    dataset = PokemonArtwork(
        resolution=cfg.resolution,
        tight_crop=cfg.tight_crop,
        include_shiny=cfg.include_shiny,
        **kwargs,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,  # a short final batch destabilises batchnorm at small batch
        persistent_workers=workers > 0,
    )


def infinite(loader: DataLoader):
    """GAN training counts steps, not epochs -- yield batches forever."""
    while True:
        yield from loader

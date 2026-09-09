"""Turn raw RGBA artwork into a square RGB training set with category labels.

    python -m src.data.preprocess --resolution 256

Three steps, each a deliberate quality lever rather than boilerplate:

  composite   Flatten alpha onto white. Training stays 3-channel -- a learned
              alpha channel is near-binary and makes a tanh generator produce
              edge halos. Transparency, if ever wanted, is recovered at
              inference by matting instead.

  tight crop  Crop to the alpha bounding box, then centre on a FRESH square
              canvas. This normalises subject scale; raw artwork frames Wailord
              and Joltik very differently. Centring on a new canvas rather than
              clamping the crop to source bounds matters -- clamping silently
              rescales the subject, reintroducing the variance being removed.

  dedupe      Requires matching shape AND colour. pHash is computed on
              luminance and is colour-blind, so a shiny hashes identically to
              its base form; pHash alone discarded 952 of 1,327 shinies, which
              is exactly the palette diversity they were included for.

The output manifest carries the taxonomy labels, so nothing downstream needs to
re-join them.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dct
from tqdm import tqdm

from src.config import (
    ARTWORK_DIR,
    FORMS_PATH,
    MANIFEST_PATH,
    META_PATH,
    TAXONOMY_PATH,
    DataConfig,
    processed_dir,
)

PHASH_IMG = 32     # DCT working size
PHASH_BITS = 8     # top-left block kept -> 64-bit hash
COLOR_GRID = 8     # RGB thumbnail side for the palette signature


def phash(img: Image.Image) -> np.uint64:
    """Perceptual hash of the luminance channel, as a packed 64-bit int."""
    small = img.convert("L").resize((PHASH_IMG, PHASH_IMG), Image.Resampling.LANCZOS)
    coeffs = dct(dct(np.asarray(small, dtype=np.float64), axis=0, norm="ortho"),
                 axis=1, norm="ortho")
    block = coeffs[:PHASH_BITS, :PHASH_BITS].flatten()
    # Exclude the DC term from the median: it dwarfs everything else and would
    # drag the threshold, flattening the hash.
    bits = block > np.median(block[1:])
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def color_signature(img: Image.Image) -> np.ndarray:
    """Coarse RGB thumbnail in [0,1] -- the colour half of the duplicate test."""
    small = img.convert("RGB").resize((COLOR_GRID, COLOR_GRID), Image.Resampling.LANCZOS)
    return np.asarray(small, dtype=np.float32).ravel() / 255.0


def _popcount64(x: np.ndarray) -> np.ndarray:
    """Vectorised Hamming weight so dedupe stays a numpy op, not a Python loop."""
    return np.unpackbits(x.astype(">u8").view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def load_and_transform(path: Path, cfg: DataConfig) -> Image.Image | None:
    """Composite, optionally tight-crop, and resize one artwork file."""
    with Image.open(path) as raw:
        img = raw.convert("RGBA")

    alpha = img.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        return None  # fully transparent, nothing to learn from

    flat = Image.new("RGB", img.size, cfg.background)
    flat.paste(img, mask=alpha)

    if cfg.tight_crop:
        flat = flat.crop(bbox)
        side = int(max(flat.size) * (1 + 2 * cfg.crop_margin))
    elif flat.width != flat.height:
        side = max(flat.size)
    else:
        side = None

    if side is not None:
        canvas = Image.new("RGB", (side, side), cfg.background)
        canvas.paste(flat, ((side - flat.width) // 2, (side - flat.height) // 2))
        flat = canvas

    return flat.resize((cfg.resolution, cfg.resolution), Image.Resampling.LANCZOS)


def find_duplicates(hashes, colors, threshold: int, color_threshold: float) -> np.ndarray:
    """Indices to drop, keeping the first member of each near-duplicate group.

    A duplicate must agree on BOTH structure and palette. Requiring both is what
    keeps shinies while still dropping genuinely redundant cosmetic forms.
    """
    n = len(hashes)
    drop = np.zeros(n, dtype=bool)
    for i in range(n):
        if drop[i]:
            continue
        rest = np.arange(i + 1, n)
        rest = rest[~drop[rest]]
        if rest.size == 0:
            continue
        same_shape = _popcount64(np.bitwise_xor(hashes[i], hashes[rest])) <= threshold
        same_color = np.abs(colors[rest] - colors[i]).mean(axis=1) <= color_threshold
        drop[rest[same_shape & same_color]] = True
    return drop


def load_labels() -> tuple[dict, dict, dict]:
    """taxonomy labels, form->species mapping, and type metadata."""
    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8")) if TAXONOMY_PATH.exists() else {}
    forms = json.loads(FORMS_PATH.read_text(encoding="utf-8")) if FORMS_PATH.exists() else {}
    types: dict[int, list[str]] = {}
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        types = {int(k): v for k, v in meta.get("types_by_id", {}).items()}
    return taxonomy, forms, types


def label_for(dex_id: int, taxonomy: dict, forms: dict) -> dict:
    """Look up a species' label, resolving alternate forms to their base species."""
    key = str(dex_id)
    if key not in taxonomy and key in forms:
        key = str(forms[key])  # a Mega/Alolan form is the same creature
    entry = taxonomy.get(key) or {}
    return {
        "top": entry.get("top") or "",       # merged -- the conditioning label
        "group": entry.get("group") or "",   # pre-merge, kept for audit
        "sub": entry.get("sub") or "",
        "species": entry.get("name") or "",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--no-tight-crop", action="store_true")
    ap.add_argument("--no-shiny", action="store_true")
    ap.add_argument("--dedupe-threshold", type=int, default=None)
    ap.add_argument("--color-threshold", type=float, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = DataConfig(
        resolution=args.resolution,
        tight_crop=not args.no_tight_crop,
        include_shiny=not args.no_shiny,
    )
    if args.dedupe_threshold is not None:
        cfg.dedupe_threshold = args.dedupe_threshold
    if args.color_threshold is not None:
        cfg.color_threshold = args.color_threshold

    if not MANIFEST_PATH.exists():
        raise SystemExit("manifest.json missing -- run `python -m src.data.download` first")
    records = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not cfg.include_shiny:
        records = [r for r in records if not r["shiny"]]
    records = [r for r in records if (ARTWORK_DIR / r["local"]).exists()]
    print(f"source: {len(records)} files -> {cfg.resolution}px, tight_crop={cfg.tight_crop}")

    def work(rec):
        img = load_and_transform(ARTWORK_DIR / rec["local"], cfg)
        if img is None:
            return rec, None, None, None
        return rec, img, phash(img), color_signature(img)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(tqdm(pool.map(work, records), total=len(records), desc="transform"))

    results = [t for t in results if t[1] is not None]
    blank = len(records) - len(results)

    hashes = np.array([h for _, _, h, _ in results], dtype=np.uint64)
    colors = np.stack([c for _, _, _, c in results])
    drop = find_duplicates(hashes, colors, cfg.dedupe_threshold, cfg.color_threshold)
    print(f"dedupe: dropping {int(drop.sum())} near-duplicates "
          f"(pHash <= {cfg.dedupe_threshold} AND colour delta <= {cfg.color_threshold})")

    taxonomy, forms, types = load_labels()
    out_dir = processed_dir(cfg.resolution, cfg.tight_crop)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (rec, img, h, _) in enumerate(tqdm(results, desc="write")):
        variant = "shiny" if rec["shiny"] else "normal"
        name = f"{variant}_{rec['dex_id']}.png"
        keep = not drop[i]
        if keep:
            img.save(out_dir / name, format="PNG", optimize=True)
        label = label_for(rec["dex_id"], taxonomy, forms)
        rows.append({
            "file": name,
            "dex_id": rec["dex_id"],
            "species": label["species"],
            "generation": rec["generation"] if rec["generation"] else "",
            "shiny": int(rec["shiny"]),
            "is_form": int(rec["is_form"]),
            "top": label["top"],
            "group": label["group"],
            "sub": label["sub"],
            "types": "|".join(types.get(rec["dex_id"], [])),
            "phash": int(h),
            "kept": int(keep),
        })

    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    kept = [r for r in rows if r["kept"]]
    kept_normal = sum(1 for r in kept if not r["shiny"])
    labelled = sum(1 for r in kept if r["top"])
    print(f"\nblank/skipped : {blank}")
    print(f"kept          : {len(kept)}  ({kept_normal} normal, {len(kept)-kept_normal} shiny)")
    print(f"labelled      : {labelled}/{len(kept)} ({100*labelled/len(kept):.0f}%)")
    print(f"output        : {out_dir}")
    print(f"manifest      : {manifest}")
    print(f"\nNote: shinies add palette diversity but zero shape diversity -- the "
          f"effective structural dataset is ~{kept_normal} images.")


if __name__ == "__main__":
    main()

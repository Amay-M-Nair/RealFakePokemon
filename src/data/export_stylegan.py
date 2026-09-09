"""Export the labelled dataset as per-class folders for StyleGAN2-ADA.

    python -m src.data.export_stylegan

Writes data/stylegan/<slug>/*.png -- one folder per class, containing only
images that carry a class label. The 112 discarded images are excluded, using
the same rule as PokemonArtwork(labelled_only=True) in src/data/dataset.py.

The images are already 256x256 RGB PNG, so this copies rather than re-encodes.

Note this deliberately stops at folders. The .zip that StyleGAN's
training/dataset.py actually consumes is built on Kaggle by NVlabs' own
`dataset_tool.py`. Hand-rolling that zip risks a silent format mismatch, and
letting their tool do it removes the possibility entirely.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from src.config import DATA_DIR, processed_dir

STYLEGAN_DIR = DATA_DIR / "stylegan"


def slugify(name: str) -> str:
    """'Plant & Fungus' -> 'plant_fungus'. Stable, filesystem-safe, readable."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return slug.strip("_")


def load_rows(resolution: int, tight_crop: bool) -> tuple[Path, list[dict]]:
    root = processed_dir(resolution, tight_crop)
    manifest = root / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(
            f"{manifest} missing -- run "
            f"`python -m src.data.preprocess --resolution {resolution}` first"
        )
    with manifest.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return root, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--no-tight-crop", action="store_true")
    ap.add_argument("--classes", nargs="*", default=None,
                    help="export only these classes (default: all)")
    ap.add_argument("--out", type=Path, default=STYLEGAN_DIR)
    ap.add_argument("--zip", action="store_true",
                    help="also write data/stylegan.zip for upload to Kaggle, "
                         "whose dataset UI does not reliably accept folders")
    args = ap.parse_args()

    root, rows = load_rows(args.resolution, not args.no_tight_crop)

    kept = [r for r in rows if r["kept"] == "1"]
    labelled = [r for r in kept if r["top"]]
    discarded = len(kept) - len(labelled)
    if args.classes:
        labelled = [r for r in labelled if r["top"] in args.classes]

    if not labelled:
        raise SystemExit("nothing to export -- check --classes against the manifest")

    by_class: dict[str, list[dict]] = {}
    for r in labelled:
        by_class.setdefault(r["top"], []).append(r)

    if args.out.exists():
        shutil.rmtree(args.out)   # stale classes must not survive a re-export
    args.out.mkdir(parents=True)

    summary = {"resolution": args.resolution, "classes": {}}
    print(f"exporting {len(labelled)} images ({discarded} discarded, excluded)\n")

    for name, items in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
        slug = slugify(name)
        dest = args.out / slug
        dest.mkdir(parents=True)
        for r in items:
            shutil.copy2(root / r["file"], dest / r["file"])
        summary["classes"][slug] = {"label": name, "count": len(items)}
        print(f"  {name:<22} {len(items):>5}  ->  {dest.relative_to(DATA_DIR.parent)}")

    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nsummary -> {args.out / 'summary.json'}")

    # Verify what actually landed on disk, rather than trusting the copy loop.
    print("\n--- verification ---")
    ok = True
    for slug, meta in summary["classes"].items():
        files = sorted((args.out / slug).glob("*.png"))
        if len(files) != meta["count"]:
            print(f"  FAIL {slug}: {len(files)} files on disk, expected {meta['count']}")
            ok = False
        sizes, modes = Counter(), Counter()
        for f in files:
            with Image.open(f) as im:
                sizes[im.size] += 1
                modes[im.mode] += 1
        if set(sizes) != {(args.resolution, args.resolution)} or set(modes) != {"RGB"}:
            print(f"  FAIL {slug}: sizes={dict(sizes)} modes={dict(modes)}")
            ok = False
    total = sum(len(list((args.out / s).glob("*.png"))) for s in summary["classes"])
    print(f"  {len(summary['classes'])} classes, {total} images, all "
          f"{args.resolution}x{args.resolution} RGB")
    print(f"  discarded images present: 0 (filtered on top != '')")

    if args.zip:
        archive = args.out.with_suffix(".zip")
        # Class folders sit at the archive root, so Kaggle's auto-extract puts
        # them directly under /kaggle/input/<dataset>/. Stored, not deflated:
        # PNGs are already compressed, so deflating costs minutes and saves ~1%.
        import zipfile

        with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
            for f in sorted(args.out.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(args.out))
        print(f"\narchive -> {archive}  ({archive.stat().st_size/2**20:.0f} MB)")

        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
        assert len(names) == total + 1, f"archive holds {len(names)}, expected {total + 1}"
        print(f"  verified: {len(names)} entries ({total} images + summary.json)")
        print(f"\nNext: upload {archive.name} as a Kaggle Dataset (it auto-extracts).")
    else:
        print(f"\nNext: re-run with --zip to produce an uploadable archive.")

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

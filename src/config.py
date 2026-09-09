"""Paths and dataset configuration.

Phase 1 only: nothing here knows about models or training.
"""

from dataclasses import dataclass
from pathlib import Path

# --- Paths -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARTWORK_DIR = RAW_DIR / "official-artwork"
PROCESSED_DIR = DATA_DIR / "processed"

MANIFEST_PATH = RAW_DIR / "manifest.json"    # what was downloaded
META_PATH = RAW_DIR / "meta.json"            # type metadata
SPECIES_PATH = RAW_DIR / "species.json"      # genus / shape / egg groups / evo chain
FORMS_PATH = RAW_DIR / "forms.json"          # alternate-form id -> base species id
BULBAPEDIA_PATH = RAW_DIR / "bulbapedia.json"  # scraped design-origin taxonomy
TAXONOMY_PATH = RAW_DIR / "taxonomy.json"    # derived category labels

# --- Dataset source ----------------------------------------------------------

SPRITES_REPO = "PokeAPI/sprites"
SPRITES_BRANCH = "master"
ARTWORK_PREFIX = "sprites/pokemon/other/official-artwork"
RAW_BASE_URL = f"https://raw.githubusercontent.com/{SPRITES_REPO}/{SPRITES_BRANCH}"
POKEAPI = "https://pokeapi.co/api/v2"

NATIONAL_DEX_MAX = 1025
# Alternate forms use synthetic ids from 10001. The real maximum today is 10326;
# the margin costs a few cheap 404s and keeps the script working as forms are
# added.
FORM_ID_MIN = 10001
FORM_ID_MAX = 10500
GEN9_START = 906

# National dex id of the last species in each generation. Form ids (>= 10000)
# have no generation of their own.
GEN_BOUNDARIES = (151, 251, 386, 493, 649, 721, 809, 905, 1025)


@dataclass
class DataConfig:
    """Preprocessing. Every field here is a lever that changed the outcome."""

    resolution: int = 256
    include_shiny: bool = True

    # Crop to the alpha bounding box before resizing, which normalises subject
    # scale across the set -- raw artwork frames Wailord and Joltik very
    # differently. Set False to build the uncropped control set.
    tight_crop: bool = True
    crop_margin: float = 0.06
    background: tuple[int, int, int] = (255, 255, 255)

    # Two images are near-duplicates only if they match on BOTH axes. pHash is
    # computed on luminance and cannot tell a shiny from its base form, so on
    # its own it discards ~72% of shinies -- exactly the palette diversity they
    # were included for.
    dedupe_threshold: int = 4        # pHash Hamming distance
    color_threshold: float = 0.02    # mean abs RGB difference, 0-1 scale

    num_workers: int = 8


def processed_dir(resolution: int, tight_crop: bool = True) -> Path:
    """Cropped and uncropped sets live side by side so they can be compared."""
    return PROCESSED_DIR / f"{resolution}{'' if tight_crop else '-uncropped'}"


def generation_of(dex_id: int) -> int | None:
    """None for alternate forms, which have no generation of their own."""
    if dex_id >= 10000:
        return None
    for gen, last in enumerate(GEN_BOUNDARIES, start=1):
        if dex_id <= last:
            return gen
    return None

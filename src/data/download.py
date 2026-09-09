"""Fetch official Pokemon artwork and the species metadata used to classify it.

    python -m src.data.download

Idempotent and cached: a second run downloads nothing. Everything lands in
data/raw/, which is gitignored -- the artwork is copyrighted and must never be
committed.

Why the file listing is obtained by probing the CDN rather than asking GitHub:
`git/trees/master?recursive=1` comes back with `truncated: true` for this repo,
which silently drops files, and the unauthenticated API allows only 60
requests/hour (easily exhausted -- a first attempt died on a 403 with a
42-minute reset). Artwork filenames are exactly `<id>.png`, so the id space can
be enumerated and probed against raw.githubusercontent.com, which has no
comparable limit. The ~384 misses return 404 and are dropped.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

from src.config import (
    ARTWORK_DIR,
    ARTWORK_PREFIX,
    FORM_ID_MAX,
    FORM_ID_MIN,
    GEN9_START,
    MANIFEST_PATH,
    META_PATH,
    NATIONAL_DEX_MAX,
    POKEAPI,
    RAW_BASE_URL,
    RAW_DIR,
    FORMS_PATH,
    SPECIES_PATH,
    generation_of,
)

# PokeAPI type slots that are not real elemental types.
NON_ELEMENTAL_TYPES = {"unknown", "shadow", "stellar"}


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "pokemon-gan/0.1 (personal research project)"})
    return s


def get_json(sess: requests.Session, url: str, retries: int = 4) -> dict:
    """Back off on rate limits rather than hammering."""
    for attempt in range(retries):
        resp = sess.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (403, 429) and attempt < retries - 1:
            time.sleep(2**attempt * 5)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"failed to fetch {url}")


# --- artwork -----------------------------------------------------------------

def record(dex_id: int, shiny: bool) -> dict:
    variant = "shiny" if shiny else "normal"
    return {
        "filename": f"{dex_id}.png",
        "dex_id": dex_id,
        "is_form": dex_id >= 10000,
        "generation": generation_of(dex_id),
        "shiny": shiny,
        "url": f"{RAW_BASE_URL}/{ARTWORK_PREFIX}/{'shiny/' if shiny else ''}{dex_id}.png",
        "local": f"{variant}/{dex_id}.png",
    }


def candidates(include_shiny: bool = True) -> list[dict]:
    ids = list(range(1, NATIONAL_DEX_MAX + 1)) + list(range(FORM_ID_MIN, FORM_ID_MAX + 1))
    variants = [False, True] if include_shiny else [False]
    return [record(i, s) for s in variants for i in ids]


def _download_one(sess: requests.Session, rec: dict, dest_root: Path) -> str:
    dest = dest_root / rec["local"]
    if dest.exists() and dest.stat().st_size > 0:
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(4):
        resp = sess.get(rec["url"], timeout=60)
        if resp.status_code == 404:
            return "missing"
        if resp.status_code in (403, 429) and attempt < 3:
            time.sleep(2**attempt * 2)
            continue
        resp.raise_for_status()
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(resp.content)
        tmp.replace(dest)  # atomic: an interrupted run never leaves a half file
        return "downloaded"
    return "missing"


def download_artwork(records: list[dict], workers: int = 12) -> tuple[list[dict], dict[str, int]]:
    """Download everything that exists; return the surviving records and counts."""
    ARTWORK_DIR.mkdir(parents=True, exist_ok=True)
    counts = {"downloaded": 0, "skipped": 0, "missing": 0}
    sessions = [session() for _ in range(workers)]
    present: list[dict] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, sessions[i % workers], rec, ARTWORK_DIR): rec
            for i, rec in enumerate(records)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="artwork"):
            status = fut.result()
            counts[status] += 1
            if status != "missing":
                present.append(futures[fut])

    present.sort(key=lambda r: (r["shiny"], r["dex_id"]))
    return present, counts


# --- metadata ----------------------------------------------------------------

def download_types(sess: requests.Session) -> dict:
    """Type metadata, fetched per-type (18 requests) not per-Pokemon (~1,300)."""
    index = get_json(sess, f"{POKEAPI}/pokemon?limit=20000")
    name_to_id = {
        r["name"]: int(r["url"].rstrip("/").rsplit("/", 1)[-1]) for r in index["results"]
    }
    by_id: dict[int, list[str]] = {}
    for tref in tqdm(get_json(sess, f"{POKEAPI}/type?limit=100")["results"], desc="types"):
        if tref["name"] in NON_ELEMENTAL_TYPES:
            continue
        for entry in get_json(sess, tref["url"])["pokemon"]:
            pid = name_to_id.get(entry["pokemon"]["name"])
            if pid is not None:
                by_id.setdefault(pid, []).append(tref["name"])
        time.sleep(0.2)
    return {
        "name_to_id": name_to_id,
        "types_by_id": {str(k): v for k, v in sorted(by_id.items())},
    }


def _species_one(sess: requests.Session, dex_id: int) -> dict | None:
    for attempt in range(4):
        r = sess.get(f"{POKEAPI}/pokemon-species/{dex_id}", timeout=30)
        if r.status_code == 404:
            return None
        if r.status_code in (403, 429) and attempt < 3:
            time.sleep(2**attempt * 2)
            continue
        r.raise_for_status()
        d = r.json()
        genus = next((g["genus"] for g in d["genera"] if g["language"]["name"] == "en"), "")
        return {
            "id": d["id"],
            "name": d["name"],
            # "Mouse Pokemon" -> "Mouse". The genus is the Pokedex's own
            # hand-written category and is what taxonomy.py classifies on.
            "genus": genus.replace("Pokémon", "").replace("Pokemon", "").strip(),
            "shape": (d.get("shape") or {}).get("name", ""),
            "egg_groups": [e["name"] for e in d.get("egg_groups", [])],
            "evolution_chain": (d.get("evolution_chain") or {}).get("url", ""),
            "is_legendary": d.get("is_legendary", False),
            "is_mythical": d.get("is_mythical", False),
        }
    return None


def _form_species(sess: requests.Session, form_id: int) -> tuple[int, int] | None:
    """Map an alternate-form id to the base species it belongs to."""
    for attempt in range(4):
        r = sess.get(f"{POKEAPI}/pokemon/{form_id}", timeout=30)
        if r.status_code == 404:
            return None
        if r.status_code in (403, 429) and attempt < 3:
            time.sleep(2**attempt * 2)
            continue
        r.raise_for_status()
        url = (r.json().get("species") or {}).get("url", "")
        return (form_id, int(url.rstrip("/").rsplit("/", 1)[-1])) if url else None
    return None


def download_forms(form_ids: list[int], workers: int = 8) -> dict[str, int]:
    """form id -> base species id.

    Alternate forms (Alolan, Mega, Gmax, ...) use synthetic ids from 10001 and
    have no species record of their own, so they cannot be classified directly.
    They inherit their base species' label instead -- which is right anyway,
    since a Mega is the same creature.
    """
    sessions = [session() for _ in range(workers)]
    out: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_form_species, sessions[i % workers], fid)
            for i, fid in enumerate(form_ids)
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="forms"):
            pair = fut.result()
            if pair:
                out[str(pair[0])] = pair[1]
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def download_species(workers: int = 8) -> dict[str, dict]:
    """One pass over every species. ~1,025 requests, cached forever after."""
    sessions = [session() for _ in range(workers)]
    out: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_species_one, sessions[i % workers], i)
            for i in range(1, NATIONAL_DEX_MAX + 1)
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="species"):
            rec = fut.result()
            if rec:
                out[rec["id"]] = rec
    return {str(k): v for k, v in sorted(out.items())}


# --- entrypoint --------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--no-shiny", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="re-fetch even if cached")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sess = session()

    print(f"probing {NATIONAL_DEX_MAX} dex ids + forms against the CDN (404s expected) ...")
    records, counts = download_artwork(candidates(not args.no_shiny), args.workers)
    MANIFEST_PATH.write_text(json.dumps(records, indent=1), encoding="utf-8")

    normal = sum(1 for r in records if not r["shiny"])
    gen9 = sum(1 for r in records if not r["shiny"] and r["dex_id"] >= GEN9_START)
    print(f"artwork: {counts}")
    print(f"  total={len(records)}  normal={normal}  shiny={len(records)-normal}  gen9={gen9}")

    if META_PATH.exists() and not args.refresh:
        print(f"types: cached at {META_PATH}")
    else:
        META_PATH.write_text(json.dumps(download_types(sess), indent=1), encoding="utf-8")
        print(f"types -> {META_PATH}")

    if SPECIES_PATH.exists() and not args.refresh:
        n = len(json.loads(SPECIES_PATH.read_text(encoding="utf-8")))
        print(f"species: cached, {n} records at {SPECIES_PATH}")
    else:
        species = download_species(args.workers)
        SPECIES_PATH.write_text(json.dumps(species, indent=1), encoding="utf-8")
        print(f"species: {len(species)} records -> {SPECIES_PATH}")

    if FORMS_PATH.exists() and not args.refresh:
        n = len(json.loads(FORMS_PATH.read_text(encoding="utf-8")))
        print(f"forms: cached, {n} mappings at {FORMS_PATH}")
    else:
        form_ids = sorted({r["dex_id"] for r in records if r["is_form"]})
        forms = download_forms(form_ids, args.workers)
        FORMS_PATH.write_text(json.dumps(forms, indent=1), encoding="utf-8")
        print(f"forms: {len(forms)}/{len(form_ids)} mapped to base species -> {FORMS_PATH}")


if __name__ == "__main__":
    main()

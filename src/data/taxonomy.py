"""Assign every Pokemon species a design-origin category.

    python -m src.data.taxonomy

The source of truth is the Bulbapedia page "Pokemon designed after animals or
plants" (see src/data/bulbapedia.py), a hand-curated mapping from each species
to the real organism it was designed after. It resolves 836 of 1,025 species.

This replaced an earlier approach that inferred categories from the Pokedex
*genus* with keyword rules. That reached only 35% coverage and was persistently
wrong in ways that were tedious to chase: the genus is a flavour label, so
"Forest" matched *ore* (Sceptile -> Minerals), "Vibration" matched *rat*
(Vibrava -> Rats), "Stag Beetle" matched *bee* (Pinsir -> Wasps) and "Starling"
matched *star* (Starly -> Aliens). Reading a curated list beats inferring one.

Two signals only:

  1. BULBAPEDIA  the curated design origin. Authoritative.
  2. LINE VOTE   a species the page omits inherits its evolution line's
                 category, if the line has one. Evolution lines share a design
                 origin, so this is safe and fills small gaps.

Species the page does not list are either rehomed into a class they visually
belong to -- two of which ("Mineral & Construct", "Amorphous & Ghost") exist
only for them -- or discarded outright. Nothing is kept as filler: a grab-bag
class teaches a generator nothing.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

from src.config import BULBAPEDIA_PATH, SPECIES_PATH, TAXONOMY_PATH

# Small classes merged into viable ones. A GAN condition needs roughly 50+
# images to learn anything; below that it produces garbage for that label.
#
# The merges are visual, not strictly phylogenetic, because the label exists to
# give the generator a coherent distribution to model:
#   - Mollusc / Echinoderm / Cnidaria are genuinely alike -- soft-bodied,
#     shelled or tentacled, no limbs.
#   - Fungi are a separate kingdom from plants, but Foongus and Morelull look
#     like plants, which is what matters here.
#   - Clitellata is the weak one: it is Onix, Steelix and Orthworm, segmented
#     rock serpents that resemble no other invertebrate. Folded in only because
#     8 images cannot support a class either way. Flagged, not hidden.
#
# Arthropod stays separate despite also being an invertebrate: it is large
# enough to stand alone, and insects look nothing like molluscs.
#
# The pre-merge group is preserved as `group`, so this is reversible.
MERGES: dict[str, str] = {
    "Mollusc": "Invertebrate",
    "Echinoderm": "Invertebrate",
    "Cnidaria": "Invertebrate",
    "Clitellata": "Invertebrate",
    "Fungus": "Plant & Fungus",
    "Plant": "Plant & Fungus",
}

# --- what happens to the species the source page does not list -----------------
#
# Two coherent classes are pulled out of that residual; everything else is
# discarded rather than kept as filler. Both were chosen after rendering them
# (docs/assets/other_subclasses.png), not from the numbers alone -- "Legendary"
# had 78 images, comfortably above the floor, and was still rejected because
# legendary is a *status*, not a look: a bird, an insect, a robot and a rock
# golem share nothing visually.
#
# Egg group drives the split. That signal was rejected earlier in this module
# for the main taxonomy (the "mineral" group put Porygon beside Geodude), but
# the failure mode does not apply here: this bucket IS the inorganic set, so
# Porygon belongs in "Mineral & Construct".
RESIDUAL_RULES: list[tuple[str, str]] = [
    ("mineral", "Mineral & Construct"),        # rock, metal, machine, construct
    ("indeterminate", "Amorphous & Ghost"),    # gas, sludge, spirit
]

# Dragons are too few to stand alone (13 images), so they join an existing
# class rather than being dropped.
#
# Flagging honestly: Reptilian is the more natural home for a serpentine dragon
# than Mammalian, and this puts Dragonair beside Vulpix. It is one word to
# change if it looks wrong in training.
DRAGON_TARGET = "Mammalian"

# Legendaries, assigned individually -- 30 species is small enough to judge one
# at a time rather than trust a rule. None means discard: no existing class
# resembles them.
LEGENDARY_ASSIGNMENTS: dict[str, str | None] = {
    # rock / metal / crystal golems and constructs
    "regirock": "Mineral & Construct",   "regice": "Mineral & Construct",
    "registeel": "Mineral & Construct",  "regieleki": "Mineral & Construct",
    "regidrago": "Mineral & Construct",  "regigigas": "Mineral & Construct",
    "melmetal": "Mineral & Construct",   "meltan": "Mineral & Construct",
    "magearna": "Mineral & Construct",   "diancie": "Mineral & Construct",
    "heatran": "Mineral & Construct",
    "darkrai": "Amorphous & Ghost",      # shadow body, no fixed silhouette
    "genesect": "Arthropod",             # a robotic beetle, but plainly a beetle
    # quadruped beasts
    "type-null": "Mammalian", "silvally": "Mammalian", "spectrier": "Mammalian",
    # dragons, following DRAGON_TARGET
    "rayquaza": DRAGON_TARGET, "zekrom": DRAGON_TARGET, "kyurem": DRAGON_TARGET,
    "reshiram": DRAGON_TARGET, "latias": DRAGON_TARGET, "latios": DRAGON_TARGET,
    # no existing class resembles these
    "hoopa": None, "jirachi": None, "deoxys": None, "volcanion": None,
    "uxie": None, "mesprit": None, "azelf": None, "eternatus": None,
}


def residual_class(rec: dict) -> str | None:
    """Where a species the page omits ends up. None means discard."""
    if rec["is_legendary"] or rec["is_mythical"]:
        return LEGENDARY_ASSIGNMENTS.get(rec["name"])
    eggs = set(rec["egg_groups"])
    for group, target in RESIDUAL_RULES:
        if group in eggs:
            return target
    if "dragon" in eggs:
        return DRAGON_TARGET
    return None   # Fairy and Misc: discarded, as agreed


# Cases that must never regress. Each was a real bug in the genus-rule era,
# as (species, field, value it must NOT have, what went wrong). The field
# matters: Pinsir IS an Arthropod -- the bug was it landing in "Bee and wasp".
REGRESSIONS = [
    ("sceptile",  "group", "Minerals",     'genus "Forest" contained "ore"'),
    ("vibrava",   "sub", "Rodent",       'genus "Vibration" contained "rat"'),
    ("pinsir",    "sub", "Bee and wasp", 'genus "Stag Beetle" matched "bee"'),
    ("starly",    "group", "Aliens",       'genus "Starling" matched "star"'),
    ("zubat",     "group", "Avian",        "bats are mammals, not birds"),
    ("magikarp",  "group", "Reptilian",    "line vote from a wrongly-tagged Gyarados"),
    ("seel",      "sub", "Feline",       'genus "Sea Lion" matched "lion"'),
]

# Spot-checks on the SCRAPE, so they name the pre-merge group. Checking the
# merged label here would conflate two separate concerns -- the merge has
# its own assertion below.
EXPECTATIONS = [
    ("bulbasaur", "Amphibian"),   # frog-based
    ("charmander", "Amphibian"),  # salamander-based -- salamanders are amphibians
    ("squirtle", "Reptilian"),
    ("pidgey", "Avian"),
    ("krabby", "Arthropod"),
    ("magikarp", "Fish"),
    ("zubat", "Mammalian"),       # bat
    ("tentacool", "Cnidaria"),
    ("staryu", "Echinoderm"),
    ("shroomish", "Fungus"),
    ("oddish", "Plant"),
    ("shellder", "Mollusc"),
]


def evolution_lines(species: dict[str, dict]) -> dict[str, list[str]]:
    chains: dict[str, list[str]] = defaultdict(list)
    for sid, rec in species.items():
        chains[rec["evolution_chain"] or f"solo:{sid}"].append(sid)
    return {sid: ids for ids in chains.values() for sid in ids}


def build(species: dict[str, dict], bulba: dict[str, dict]) -> dict[str, dict]:
    by_name = {rec["name"]: sid for sid, rec in species.items()}
    line_of = evolution_lines(species)
    labels: dict[str, dict] = {}

    # 1. curated design origin
    for slug, entry in bulba.items():
        sid = by_name.get(slug)
        if sid:
            labels[sid] = {"top": entry["top"], "sub": entry["sub"], "why": "bulbapedia"}

    # 2. fill within-line gaps -- evolution lines share a design origin
    for sid in species:
        if sid in labels:
            continue
        known = [labels[m] for m in line_of.get(sid, []) if m in labels]
        if known:
            top = Counter(k["top"] for k in known).most_common(1)[0][0]
            sub = Counter(k["sub"] for k in known if k["top"] == top).most_common(1)[0][0]
            labels[sid] = {"top": top, "sub": sub, "why": "line-vote"}

    for sid, rec in species.items():
        if sid in labels:
            entry, group = labels[sid], labels[sid]["top"]
        else:
            target = residual_class(rec)
            entry = {"top": target, "sub": "", "why": "residual" if target else "discarded"}
            group = None
        entry.update(
            name=rec["name"], genus=rec["genus"], shape=rec["shape"],
            group=group or "",                             # pre-merge, for audit
            top=MERGES.get(group, group) if group else entry["top"],
        )
        labels[sid] = entry
    return labels


def verify(labels: dict[str, dict]) -> bool:
    by_name = {v["name"]: v for v in labels.values()}
    ok = True
    print("\n--- verification ---")

    unassigned = [v["name"] for v in labels.values() if not v["top"] and v["why"] != "discarded"]
    if unassigned:
        print(f"  FAIL residual: {len(unassigned)} species have no class and no discard reason")
        ok = False
    stray = [n for n, t in LEGENDARY_ASSIGNMENTS.items() if t and (by_name.get(n) or {}).get("top") != t]
    if stray:
        print(f"  FAIL legendary: {stray} did not land where assigned")
        ok = False
    print(f"  residual: every species is either classed or explicitly discarded")
    print(f"  legendary: {sum(1 for t in LEGENDARY_ASSIGNMENTS.values() if t)} assigned, "
          f"{sum(1 for t in LEGENDARY_ASSIGNMENTS.values() if not t)} discarded, all verified")

    for name, expected in EXPECTATIONS:
        got = (by_name.get(name) or {}).get("group")
        if got != expected:
            print(f"  FAIL expectation: {name} has group {got}, expected {expected}")
            ok = False
    print(f"  expectations: {len(EXPECTATIONS)} spot-checks on the scrape "
          f"(pre-merge groups)")

    for name, field, forbidden, reason in REGRESSIONS:
        entry = by_name.get(name)
        if entry and entry[field] == forbidden:
            print(f"  FAIL regression: {name} {field}={forbidden} ({reason})")
            ok = False
    print(f"  regressions: {len(REGRESSIONS)} historical bugs all absent")

    for source, target in MERGES.items():
        stragglers = [v["name"] for v in labels.values() if v["group"] == source and v["top"] != target]
        if stragglers:
            print(f"  FAIL merge: {source} -> {target} missed {len(stragglers)} ({stragglers[:3]})")
            ok = False
        if not any(v["top"] == source for v in labels.values()):
            continue
        print(f"  FAIL merge: {source} still present as a top-level class")
        ok = False
    print(f"  merges: {len(set(MERGES.values()))} clusters applied, no source class survives")
    return ok


def report(labels: dict[str, dict], sample: int = 20) -> None:
    total = len(labels)
    done = sum(1 for v in labels.values() if v["why"] != "residual")
    classed = sum(1 for v in labels.values() if v["top"])
    print(f"\ncurated source covers {done}/{total} species ({100*done/total:.0f}%); "
          f"{classed-done} rehomed, {total-classed} discarded\n")

    by_top = defaultdict(Counter)
    for v in labels.values():
        if v["top"]:
            by_top[v["top"]][v["group"]] += 1
    for top, groups in sorted(by_top.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(groups.values())
        merged = [g for g in groups if g and g != top]
        note = f"   <- merged from {', '.join(sorted(merged))}" if merged else ""
        print(f"  {top:<16} {n:>4} species{note}")

    print(f"\n  evidence: {dict(Counter(v['why'] for v in labels.values() if v['top']))}")

    random.seed(0)
    picks = random.sample([v for v in labels.values() if v["top"]], min(sample, done))
    print(f"\n--- random sample of {len(picks)}, read these ---")
    for v in picks:
        print(f"  {v['name']:<14} {v['genus']:<15} -> {v['top']}/{v['sub']} ({v['why']})")

    residual = [v for v in labels.values() if v["why"] == "residual"]
    print(f"\n--- rehomed: {len(residual)} species from outside the source page ---")
    for top, n in Counter(v["top"] for v in residual).most_common():
        ex = [v["name"] for v in residual if v["top"] == top][:5]
        print(f"  -> {top:<22} {n:>3}   e.g. {', '.join(ex)}")

    dropped = [v for v in labels.values() if v["why"] == "discarded"]
    print(f"\n--- discarded: {len(dropped)} species with no coherent home ---")
    for shape, n in Counter(v["shape"] for v in dropped).most_common(5):
        ex = [v["name"] for v in dropped if v["shape"] == shape][:4]
        print(f"  shape={shape:<12} x{n:<4} e.g. {', '.join(ex)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list-unclassified", action="store_true")
    args = ap.parse_args()

    for path, hint in ((SPECIES_PATH, "src.data.download"), (BULBAPEDIA_PATH, "src.data.bulbapedia")):
        if not path.exists():
            raise SystemExit(f"{path.name} missing -- run `python -m {hint}` first")

    species = json.loads(SPECIES_PATH.read_text(encoding="utf-8"))
    bulba = json.loads(BULBAPEDIA_PATH.read_text(encoding="utf-8"))

    unresolved = sorted(set(bulba) - {r["name"] for r in species.values()})
    if unresolved:
        print(f"note: {len(unresolved)} page entries are not in PokeAPI yet "
              f"({', '.join(unresolved)}) -- newer than the cached species data")

    labels = build(species, bulba)
    TAXONOMY_PATH.write_text(json.dumps(labels, indent=1), encoding="utf-8")
    report(labels)
    ok = verify(labels)

    if args.list_unclassified:
        print("\n--- every unclassified species ---")
        for v in sorted((v for v in labels.values() if not v["top"]), key=lambda x: x["name"]):
            print(f"  {v['name']:<16} {v['genus']:<18} shape={v['shape']}")

    print(f"\nwrote {TAXONOMY_PATH}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

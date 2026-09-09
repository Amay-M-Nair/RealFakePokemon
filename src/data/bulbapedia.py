"""Scrape the Bulbapedia "Pokemon designed after animals or plants" taxonomy.

    python -m src.data.bulbapedia

This replaces guesswork. The page is a hand-curated mapping from every
distinctly-identifiable Pokemon to the real animal or plant it was designed
after, structured as:

    == {{wp|Amphibian}} Pokemon ==          <- top-level group
    === {{wp|Frog}}- and {{wp|toad}}-based === <- leaf category
    ! {{p|Bulbasaur}}                       <- the members

Reading it directly is strictly better than inferring categories from the
Pokedex genus: the genus is a flavour label ("Seed", "Dopey", "Vibration") that
often says nothing about anatomy, whereas this page is explicitly about design
origin.

Note the page only covers Pokemon "designed after animals or plants", so
inorganic designs -- Magnemite, Voltorb, Klink, most ghosts, most legendaries --
are legitimately absent and stay unclassified.

Fetched once via the MediaWiki API and cached to data/raw/bulbapedia.json.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict

import requests

from src.config import BULBAPEDIA_PATH, META_PATH

API = "https://bulbapedia.bulbagarden.net/w/api.php"
PAGE = "User:Reshi643/Pokémon designed after animals or plants"

# {{p|Name}} and {{p|Name|Display}} both link a species.
POKEMON_TEMPLATE = re.compile(r"\{\{p\|([^}|]+)(?:\|[^}]*)?\}\}")
HEADING = re.compile(r"^(={2,4})\s*(.+?)\s*\1\s*$", re.M)
# Strip {{wp|Article}} / {{wp|Article|Display}} down to the display text.
WP_TEMPLATE = re.compile(r"\{\{wp\|(?:[^}|]*\|)?([^}|]+)\}\}")

# PokeAPI slugs for names that do not transliterate mechanically.
NAME_OVERRIDES = {
    "nidoran♀": "nidoran-f",
    "nidoran♂": "nidoran-m",
    "farfetch'd": "farfetchd",
    "sirfetch'd": "sirfetchd",
    "mr. mime": "mr-mime",
    "mr. rime": "mr-rime",
    "mime jr.": "mime-jr",
    "type: null": "type-null",
    "porygon-z": "porygon-z",
    "flabébé": "flabebe",
    "ho-oh": "ho-oh",
}


def clean_heading(text: str) -> str:
    """'{{wp|Frog}}- and {{wp|toad}}-based' -> 'Frog- and toad-based'."""
    text = WP_TEMPLATE.sub(r"\1", text)
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]|]+)\]\]", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def normalise_name(name: str) -> str:
    """Bulbapedia display name -> PokeAPI slug."""
    n = name.strip().lower()
    if n in NAME_OVERRIDES:
        return NAME_OVERRIDES[n]
    n = n.replace("é", "e").replace("♀", "-f").replace("♂", "-m")
    n = n.replace(".", "").replace("'", "").replace(":", "")
    n = re.sub(r"\s+", "-", n)
    return unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()


def fetch_wikitext() -> str:
    sess = requests.Session()
    sess.headers.update({"User-Agent": "pokemon-gan/0.1 (personal research project)"})
    r = sess.get(
        API,
        params={"action": "parse", "page": PAGE, "prop": "wikitext",
                "format": "json", "formatversion": "2"},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise SystemExit(f"MediaWiki error: {data['error']}")
    return data["parse"]["wikitext"]


def parse(wikitext: str) -> dict[str, dict]:
    """name slug -> {top, sub}. The deepest heading wins as the leaf."""
    headings = [(m.start(), len(m.group(1)), clean_heading(m.group(2)))
                for m in HEADING.finditer(wikitext)]
    headings.append((len(wikitext), 2, "<end>"))

    out: dict[str, dict] = {}
    top = sub = None
    for i, (pos, level, title) in enumerate(headings[:-1]):
        if level == 2:
            top, sub = title.replace(" Pokémon", "").replace(" Pokemon", "").strip(), None
            continue
        # level 3 or 4: a leaf. Level 4 (e.g. Canine > Dog) overrides level 3.
        sub = title.replace("-based", "").replace("- and ", "/").replace("-", " ").strip()
        body = wikitext[pos:headings[i + 1][0]]
        for raw in POKEMON_TEMPLATE.findall(body):
            slug = normalise_name(raw)
            # First assignment wins, so a level-3 group does not overwrite the
            # more specific level-4 one that follows it.
            out.setdefault(slug, {"top": top, "sub": sub, "display": raw.strip()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    if BULBAPEDIA_PATH.exists() and not args.refresh:
        data = json.loads(BULBAPEDIA_PATH.read_text(encoding="utf-8"))
        print(f"cached: {len(data)} entries at {BULBAPEDIA_PATH}")
    else:
        wt = fetch_wikitext()
        print(f"fetched {len(wt):,} chars of wikitext")
        data = parse(wt)
        BULBAPEDIA_PATH.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"parsed {len(data)} species -> {BULBAPEDIA_PATH}")

    tops = Counter(v["top"] for v in data.values())
    print(f"\n{len(tops)} top-level groups:")
    for t, n in tops.most_common():
        subs = len({v["sub"] for v in data.values() if v["top"] == t})
        print(f"  {t:<16} {n:>4} species across {subs:>2} leaves")

    # how many resolve against PokeAPI's names
    if META_PATH.exists():
        name_to_id = json.loads(META_PATH.read_text(encoding="utf-8"))["name_to_id"]
        hit = [s for s in data if s in name_to_id]
        miss = sorted(set(data) - set(name_to_id))
        print(f"\nname resolution: {len(hit)}/{len(data)} matched PokeAPI slugs")
        if miss:
            print(f"  unmatched ({len(miss)}): {', '.join(miss[:25])}"
                  + (" ..." if len(miss) > 25 else ""))


if __name__ == "__main__":
    main()

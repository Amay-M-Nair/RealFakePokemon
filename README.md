# Pokémon GAN — Phase 1: Data & Classification

A dataset of official Pokémon artwork, preprocessed for GAN training and
labelled by design origin. **This phase produces data only** — no model, no
training loop, no server. Those come next, once the data is right.

## Run it

```bash
pip install -r requirements.txt

python -m src.data.download      # artwork + species metadata  (cached, idempotent)
python -m src.data.bulbapedia    # scrape the design-origin taxonomy
python -m src.data.taxonomy      # species -> category, with verification
python -m src.data.preprocess --resolution 256
```

Then open `notebooks/00_data.ipynb` for coverage, class balance, and a montage
per category.

## What comes out

`data/processed/256/manifest.csv`, one row per image:

| column | |
|---|---|
| `file` | image filename in the same directory |
| `dex_id` `species` `generation` | identity |
| `shiny` `is_form` | variant flags |
| `top` | final category — **the conditioning label** (e.g. `Mammalian`); empty if discarded |
| `group` | pre-merge category, kept so merges stay auditable |
| `sub` | leaf design origin (e.g. `Fox`, `Jellyfish`) |
| `types` | elemental types, pipe-separated |
| `phash` `kept` | dedupe bookkeeping |

| | |
|---|---|
| Downloaded | 2,666 (1,339 normal + 1,327 shiny), 434 Gen 9 |
| After dedupe | **2,231** (1,308 normal + 923 shiny) |
| **Distinct shapes** | **1,308** |
| Trainable | 2,119 / 2,231 (95%) across 10 classes |
| Resolution | 256×256 RGB, tight-cropped, white ground |

**Reason about 1,308, not 2,231.** Shinies are recolours: they add palette
diversity and zero shape diversity.

## Decisions that mattered

**The listing is obtained by probing the CDN, not the GitHub API.**
`git/trees/master?recursive=1` returns `truncated: true` for the sprites repo,
silently dropping files, and the unauthenticated API allows 60 requests/hour
(a first attempt died on a 403 with a 42-minute reset). Artwork filenames are
exactly `<id>.png`, so the id space is enumerated and probed against
`raw.githubusercontent.com`. The ~384 misses cost nothing.

**Alpha is composited onto white, not learned.** Alpha here is near-binary and
makes a `tanh` generator produce edge halos.

**Tight crop, centred on a fresh canvas.** Normalises subject scale — raw
artwork frames Wailord and Joltik very differently. Centring on a *new* canvas
rather than clamping the crop to source bounds matters; clamping silently
rescales the subject. Measured: ink centroid lands 0.1 px, 1.0 px off centre.

**Dedupe requires matching shape AND colour.** pHash runs on luminance and is
colour-blind, and a shiny is structurally identical to its base form — pHash
alone discarded **952 of 1,327 shinies**, exactly the palette diversity they
were included for. Requiring pHash Hamming ≤ 4 *and* mean RGB difference ≤ 0.02
drops 435 genuine duplicates and keeps 923 shinies.

**Classification reads a curated list instead of inferring one.** The first
attempt derived categories from the Pokédex *genus* with keyword rules and
reached 35% coverage while being persistently wrong: the genus is a flavour
label, so `"Forest"` matched *ore* (Sceptile → Minerals), `"Vibration"` matched
*rat* (Vibrava → Rats), `"Stag Beetle"` matched *bee* (Pinsir → Wasps), and
`"Starling"` matched *star* (Starly → Aliens). Scraping Bulbapedia's
hand-curated "designed after animals or plants" page gives **82%** coverage and
is right by construction.

Species the page omits are either **rehomed** into a class they visually belong
to, or **discarded**. Nothing is kept as filler — a grab-bag class teaches a
generator nothing. Two classes exist only for rehomed species:

- **Mineral & Construct** — rock, metal, machine (Magnemite, Klink, the Regis).
- **Amorphous & Ghost** — gas, sludge, spirit (Gastly, Grimer, Chandelure).

Egg group drives that split. It was rejected as a signal for the main taxonomy
(the "mineral" group put Porygon beside Geodude), but the failure mode does not
apply here: this residual *is* the inorganic set, so Porygon genuinely belongs
in Mineral & Construct. That was confirmed by rendering the groups, not by the
reasoning sounding right.

## Class balance

Every retained class clears the ~50-image floor a GAN condition needs; below
that it produces garbage for that label.

| category | images | |
|---|---|---|
| Mammalian | 789 | includes dragons |
| Arthropod | 237 | |
| Plant & Fungus | 197 | merged from Plant + Fungus |
| Reptilian | 194 | |
| Avian | 187 | |
| Mineral & Construct | 154 | rehomed from outside the source page |
| Fish | 103 | |
| Amorphous & Ghost | 96 | rehomed from outside the source page |
| Invertebrate | 91 | merged from Mollusc + Echinoderm + Cnidaria + Clitellata |
| Amphibian | 71 | |
| *(discarded)* | 112 | no coherent home |

**10 classes, 2,119 trainable images, smallest 71.**

Merges are visual rather than strictly phylogenetic, because the label exists to
give the generator a coherent distribution to model:

- Molluscs, echinoderms and cnidarians are genuinely alike — soft-bodied,
  shelled or tentacled, no limbs.
- Fungi are a separate kingdom from plants, but Foongus and Morelull *look*
  like plants.
- Arthropod stays separate despite also being an invertebrate — large enough to
  stand alone, and insects look nothing like molluscs.

### Judgement calls worth knowing about

- **Legendary was rejected as a class despite having 78 images** — comfortably
  above the floor. Legendary is a *status*, not a look: a bird, an insect, a
  robot and a rock golem share nothing visually. The 30 legendaries were instead
  assigned individually (`LEGENDARY_ASSIGNMENTS`), with 8 discarded.
- **Dragons sit in Mammalian.** They are 13 images, too few to stand alone.
  Reptilian is arguably the more natural home for a serpentine dragon; this puts
  Dragonair beside Vulpix. One constant (`DRAGON_TARGET`) changes it.
- **Clitellata → Invertebrate is the weakest merge.** It is Onix, Steelix and
  Orthworm — segmented rock serpents resembling no other invertebrate. Folded in
  only because 8 images cannot support a class either way.

The pre-merge group is preserved in the manifest's `group` column, so every
merge is reversible and auditable.

## Verification

`python -m src.data.taxonomy` exits non-zero if any check fails:

- 12 spot-checks on the scrape, one per source group (Charmander → Amphibian,
  because it is salamander-based and salamanders are amphibians).
- 7 regressions from the genus-rule era, each asserted absent on the specific
  field that was wrong — Pinsir *is* an Arthropod; the bug was "Bee and wasp".
- Every merge applied and no source class surviving.
- Every legendary landing where it was explicitly assigned.
- Every species either classed or explicitly discarded — never silently
  unlabelled.

`python -m src.data.download` is idempotent: a second run reports
`downloaded: 0, skipped: 2666`.

## Legal

Pokémon artwork is © Nintendo / Creatures Inc. / GAME FREAK. **`data/` is
gitignored and must never be committed** — use `download.py`. Non-commercial
research use only.

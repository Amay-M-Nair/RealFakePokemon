# StyleGAN2-ADA runbook

Fine-tuning NVIDIA's StyleGAN2-ADA per class, on Kaggle. This is the operational
reference; the phase plan lives in the project roadmap.

## Why Kaggle and not the local GPU

NVlabs' own spec puts StyleGAN2-ADA at 256×256 at **~5.0 GB VRAM**. The RTX 3050
has 4 GB, less ~0.7 GB for the Windows desktop. **This is a wall, not a tuning
problem** — batch size and gradient accumulation do not get under it.

Kaggle also sidesteps building the custom CUDA ops with MSVC on Windows, which
on Linux simply works.

| | |
|---|---|
| Hardware | Kaggle **T4 x2** (2 x 16 GB) or **P100** (16 GB) - either works |
| Quota | ~30 GPU-h/week, sessions capped at 12 h |
| Resolution | 256×256 |
| Source net | `lsundog-res256-paper256-kimg100000-noaug.pkl` (282 MB) |

---

## Phase 2 — harness

```bash
python -m src.data.export_stylegan --zip    # per-class folders + data/stylegan.zip
```

Then upload **`data/stylegan.zip`** as a Kaggle Dataset (its UI does not
reliably accept folders; the zip auto-extracts with the class folders at the
root) and run `notebooks/01_stylegan_setup.ipynb` with **Accelerator = GPU T4 x2
or P100** and **Internet = On**.

### If the accelerator will not start

P100 and T4 x2 share one pool, so P100 is often simply busy.

1. **Switch to `GPU T4 x2`.** A single T4 has the same 16 GB as a P100, and two
   of them is more throughput. The notebook detects the GPU count and passes
   `--gpus` accordingly, so nothing needs changing.
2. **Phone-verify the account.** Verification gates *both* GPU access and
   Internet, and Internet is required to clone NVlabs and fetch the source net.
   VoIP numbers are commonly rejected.
3. **Check the weekly quota** (~30 GPU-h/week, resets weekly).

The notebook trains one class for 20 kimg to confirm the pipeline works and to
measure speed. Six code cells, ~150 lines.

### Five upstream patches, in three files

NVlabs targets PyTorch 1.7-1.10; Kaggle ships 2.x. Five small incompatibilities
need fixing, and nothing else:

| file | problem | symptom |
|---|---|---|
| `custom_ops.py` | discards `cpp_extension.load()`'s return value | kernels report *"Failed!"* **after** compiling fine, then silently fall back to slow reference code |
| `custom_ops.py` | then re-imports by name, unsupported on modern torch | (same) |
| `misc.py` | `Sampler.__init__` no longer takes `data_source` | `TypeError: object.__init__() takes exactly one argument` |
| `grid_sample_gradfix.py` | version gate disables it on torch 2.x | `RuntimeError: derivative for aten::grid_sampler_2d_backward is not implemented`, from `.backward()` at `loss.py:131` |
| `grid_sample_gradfix.py` | `aten::grid_sampler_2d_backward` gained a 7th arg (`output_mask`) | would fail once the gate is opened |

The `patch()` helper **asserts its target string is present**. A silent no-op is
the worst outcome available here: the file looks pristine, and the failure
surfaces much later somewhere unrelated.

### grid_sample_gradfix is REQUIRED. conv2d_gradfix is not.

These two files look interchangeable and are not. Both gate themselves to torch
1.7-1.9 and warn *"Falling back to torch.nn.functional..."* on anything newer,
but only one of those fallbacks is safe.

**`grid_sample_gradfix` must be enabled.** It exists to supply the *second*
derivative of `grid_sample`, which PyTorch **still does not have**. The R1
gradient penalty differentiates through the ADA augment pipeline — which
geometrically transforms via `grid_sample` — with `create_graph=True`, then
calls `.backward()`. That is a double backward. Without the patch, training runs
normally through dataset build, network construction and the sampler, then dies
partway into the first tick.

The op it wraps, `aten::grid_sampler_2d_backward`, **still exists**; only its
signature drifted. Two patches restore it: open the gate, and reach it through
`torch.ops.aten` (rather than `torch._C._jit_get_operation`, which now returns a
`(op, overload_names)` tuple) with the extra `output_mask` argument.

**`conv2d_gradfix` must stay disabled.** Its op,
`aten::cudnn_convolution_backward_weight`, was **deleted** from PyTorch in 1.13.
There is nothing to re-point it at. Forcing that gate open raises
`TypeError: 'tuple' object is not callable` inside `backward()`. Plain
`F.conv2d` supports double backward perfectly well; the only loss is that
`no_weight_gradients()` becomes a no-op, so the R1 pass computes weight
gradients it discards. That is a modest speed cost, not a correctness one.

Both facts were verified by running the code, not inferred:

```
F.grid_sample double backward: FAILS -> derivative ... is not implemented
F.conv2d     double backward: WORKS
```

Step 2 of the notebook runs that exact double-backward on a 16x16 tensor. It
costs a fraction of a second and fails immediately if a patch did not take,
instead of ten minutes into training.

### The one check that needs a human

Step 3 renders samples straight from the pretrained pickle, before any training.
**They must look like dogs.** If they are noise, `--resume` did not load and the
run would silently train from scratch. *(Confirmed working.)*

### Verified against the upstream repo

Checked against `NVlabs/stylegan2-ada-pytorch@main` rather than assumed:

- All five patch strings match upstream `@main`, and the patched files still parse.
- `dataset_tool.py` takes `--source` / `--dest`.
- `ImageFolderDataset(path, **super_kwargs)` accepts `use_labels`, `max_size`,
  `xflip`, and exposes `image_shape`.
- Every `train.py` flag used exists, and `paper256` is a valid `--cfg` choice.

**The notebook cannot be run locally** -- 256px needs ~5 GB VRAM and the 4 GB
card cannot host it.

### A note on scope

An earlier version of this notebook was ~250 lines with extensive assertions and
defensive patching. Most of that guarded against problems it had itself created:
patching the gradfix version gate broke a working fallback, and the machinery
added to detect the breakage was pure overhead. The current version changes only
what is actually broken.

---

## Measured throughput

Fill in from the Phase 2 smoke run. Phase 3 sizes every budget from this.

| | |
|---|---|
| sec / kimg (256px, batch from `paper256`) | *pending* |
| accelerator used | *pending* |
| 500 kimg | *pending* |
| CUDA ops compiled or fell back | *pending* |

---

## Phase 3 — per-class fine-tuning

```bash
python train.py \
  --outdir=/kaggle/working/out --data=<class>.zip --gpus=<1 or 2> \
  --cfg=paper256 --mirror=1 --aug=ada --target=0.6 \
  --resume=<source or last snapshot> \
  --freezed=<tune> --snap=10 --metrics=kid50k_full --kimg=<budget>
```

### Flags that matter

- **`--cfg=paper256` is load-bearing.** The source net is literally named
  `paper256`; any other config changes layer shapes and `--resume` fails.
- `--mirror=1` doubles effective data for free — Pokémon artwork has no
  meaningful chirality.
- `--aug=ada` is the entire reason for choosing this model over plain StyleGAN2.
- `--freezed` (FreezeD) freezes the first N discriminator layers. Treat it as
  the first knob to tune, not a fixed value.
- **KID, not FID.** Consistent with Phase 1 and correct at 71–789 real images,
  where FID's covariance estimate is badly biased. Note `kid50k_full` still
  generates 50k samples, so it is expensive on a short run — use
  `--metrics=none` for smoke tests.

### The resume caveat

`--resume` restores G, D, G_ema and the ADA augment pipeline, but **not the
optimizer state**. Across Kaggle's 12 h cap that means Adam momentum resets at
every session boundary. Size each class to finish inside one session.

### Class sizes

| class | images | | class | images |
|---|---|---|---|---|
| Mammalian | 789 | | Mineral & Construct | 154 |
| Arthropod | 237 | | Fish | 103 |
| Plant & Fungus | 197 | | Amorphous & Ghost | 96 |
| Reptilian | 194 | | Invertebrate | 91 |
| Avian | 187 | | Amphibian | 71 |

Pilot order is Mammalian → Arthropod → Plant & Fungus: the three largest, and
together **58% of the 400-image target**, so failure is cheap and success
delivers most of the deliverable.

---

## Expectations

Transfer learning at 100-shot scale is a demonstrated regime (Obama, Grumpy Cat,
Panda benchmarks). But those targets are *homogeneous* — one person, one cat.
These classes are not: Amphibian's 71 images span frogs and salamanders in
varied poses.

Expect quality to track class **coherence**, not just class size. Mammalian
(789) should produce genuinely good creatures; Amphibian is the one most likely
to disappoint.

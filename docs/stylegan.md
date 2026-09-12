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

### Seven upstream patches, in four files

NVlabs targets PyTorch 1.7-1.10; Kaggle ships 2.x. Seven small incompatibilities
need fixing, and nothing else. The last two matter only for multi-GPU:

| file | problem | symptom |
|---|---|---|
| `custom_ops.py` | discards `cpp_extension.load()`'s return value | kernels report *"Failed!"* **after** compiling fine, then silently fall back to slow reference code |
| `custom_ops.py` | then re-imports by name, unsupported on modern torch | (same) |
| `misc.py` | `Sampler.__init__` no longer takes `data_source` | `TypeError: object.__init__() takes exactly one argument` |
| `grid_sample_gradfix.py` | version gate disables it on torch 2.x | `RuntimeError: derivative for aten::grid_sampler_2d_backward is not implemented`, from `.backward()` at `loss.py:131` |
| `grid_sample_gradfix.py` | `aten::grid_sampler_2d_backward` gained a 7th arg (`output_mask`) | would fail once the gate is opened |
| `train.py` | `batch_gpu` hard-coded to `mb // ref_gpus` | 8 accumulation rounds, 3 GB of 16 used |
| `training_loop.py` | ranks disagree on `noise_const` | `AssertionError: Generator.synthesis.b4.conv1.noise_const` |

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

### The multi-GPU buffer bug

`--gpus=2` fails at the first snapshot with:

```
AssertionError: Generator.synthesis.b4.conv1.noise_const
  training_loop.py:360  misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
```

Two independent causes, both real:

1. **`training_loop.py:155`** -- `if (resume_pkl is not None) and (rank == 0)`.
   Only rank 0 loads the pretrained net, so only rank 0 gets the pickle's
   `noise_const`. Rank 1 keeps the one it randomly initialised, from a different
   per-rank seed (`training_loop.py:125`).
2. **`training_loop.py:187`** -- DDP is constructed with
   `broadcast_buffers=False`. On torch 1.7-1.9 that only controlled per-forward
   syncing and buffers were *always* broadcast at construction. Modern torch
   passes the flag through to `_sync_module_states`, which then skips
   `named_buffers()` entirely. That is why this code worked in 2021.

**Parameters are still synced** -- `_sync_module_states` adds
`named_parameters()` unconditionally -- so rank 1 was never training from random
weights. Only buffers drift, and `noise_const` is read only under
`noise_mode='const'` (inference), never during training. The assert is correct
that the ranks disagree; the consequence was narrower than it looks.

Fix: broadcast params and buffers once from rank 0, after the resume block and
before DDP wraps anything. No per-iteration cost.

It also calls `torch.cuda.set_device(device)` first. **NVlabs never calls
`set_device` anywhere** -- it relies on `device_ids=[device]` when constructing
DDP. Since this broadcast runs *before* DDP exists, rank 1's tensors are on
`cuda:1` while the current device is still `cuda:0`, and NCCL in that state can
**hang rather than raise**. A hang burns quota silently, which is worse than a
crash.

`Grad strides do not match bucket view strides` also appears under DDP. It is a
gradient-layout hint, not an error, and it is filtered from the notebook output.

### Reading the training log

`conv2d_gradfix` warns on **every** convolution. Thousands of identical lines
bury the progress ticks, and would bury a real traceback too -- the R1 failure
above was diagnosed from a line number because the exception itself had been
scrolled away. The notebook filters that one string and streams the rest live.

**`Exiting...` means success.** It appears exactly twice in the upstream repo,
`training_loop.py:419` and `calc_metrics.py:72`, both only after a run finishes
normally, and `--metrics=none` means the second never executes. If the log ends
there, the run completed.

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

Smoke run, Arthropod (237 images), 20 kimg, **2026-09-10**.

| | |
|---|---|
| Environment | Kaggle, PyTorch **2.10.0+cu128**, Tesla T4 |
| GPUs used | **1** of 2 |
| CUDA ops | **compiled** (`bias_act`, `upfirdn2d`, 44 s build) |
| sec/kimg, training only | **70.9** (ticks 1-5, dead flat) |
| sec/kimg, wall clock | **75.0** (includes ~16.6 s/tick maintenance) |
| Total | 20 kimg in 26.8 min |
| GPU memory | **3.0 GB of 16** |
| Drift from source net | 0.688 |

Tick 0 reads 744 sec/kimg — kernel compilation and warm-up. It is excluded from
the figure above, which is measured between ticks.

That 75.0 is the **single-GPU baseline** and is superseded -- see the next
section. Phase 3 plans against **33.2 sec/kimg**:

| kimg | per class | 10 classes |
|---|---|---|
| 200 | 1.8 h | 18 h |
| 300 | **2.8 h** | **28 h** |
| 400 | 3.7 h | 37 h |

Against a **~30 GPU-h/week quota**, ten classes at 300 kimg fit one week, and the
**12 h session cap** holds ~1300 kimg -- four classes per session.

### Throughput, all three configs measured

`train.py:191` pins `batch_gpu = spec.mb // spec.ref_gpus`, and `ref_gpus=8` is
NVlabs' reference rig, not the local GPU count -- line 167 overrides it only
under `--cfg=auto`. So batch 64 ran as 8 sequential rounds of 8 images on one
card, at 3.0 GB of 16, with the second T4 idle.

`batch_size` stays 64 in every row, so gradients and all hyperparameters are
identical. Only the chunking and the GPU count change.

| config | rounds | sec/kimg | | 300 kimg/class | 10 classes |
|---|---|---|---|---|---|
| 1 GPU, `batch_gpu=8` | 8 | 70.9 | baseline | 5.9 h | 59 h |
| 1 GPU, `batch_gpu=32` | 2 | 65.5 | 1.08x | 5.5 h | 55 h |
| **2 GPUs, `batch_gpu=32`** | 1 | **33.2** | **2.14x** | **2.8 h** | **28 h** |

**The accumulation-round theory was mostly wrong.** Collapsing 8 rounds to 2 on
one GPU bought only 8%, because the T4 was already compute-bound at batch 8 --
the launch overhead was never the bottleneck. The second GPU was the whole win,
and 2.14x exceeds a clean 2x only because it stacks both effects.

This is what makes the full run affordable: ten classes at 300 kimg is **28 h**,
inside a single ~30 GPU-h week, against 59 h on the original config. At 2.8 h per
class the 12 h session cap fits four classes per session.

Enabling it needs two patches beyond the correctness five -- `BATCH_GPU` and the
`training_loop.py` buffer sync -- both documented above.

---

## Phase 3 — per-class fine-tuning

`notebooks/02_finetune.ipynb`. **Two cells per class** -- one to train, one to
show results -- so a class can be run and judged before the next is started.

Every class trains from the LSUN Dog net with its own dataset zip and its own
output directory. `train(cls, resume=...)` takes the checkpoint per call, so
extending one class cannot seed another. Nothing is shared between classes.

Runs on **2 GPUs with `batch_gpu=32`**, measured at 33.2 sec/kimg -- 2.14x the
single-GPU baseline. That needs all seven patches, including the two multi-GPU
ones.

```bash
python train.py   --outdir=<out> --data=<class>.zip --gpus=1   --cfg=paper256 --mirror=1 --aug=ada --target=0.6   --resume=<source net or previous snapshot>   --snap=10 --metrics=none --kimg=300
```

### Budget

300 kimg at the measured **33.2 sec/kimg** (2 GPUs, `batch_gpu=32`) is **2.8 h**,
well inside the 12 h session cap -- four classes fit one session. `kimg` counts
images *shown*, not epochs, so Mammalian's 789 images cost exactly the same as
Arthropod's 237.

| # | class | images | share of the 400-image target |
|---|---|---|---|
| 1 | `mammalian` | 789 | 37% |
| 2 | `arthropod` | 237 | 11% |
| 3 | `plant_fungus` | 197 | 9% |

All three in **one 8.3 h session**. **Stop there and judge** before spending the
remaining seven, which now cost only ~19 h more.

### Run it as a saved version

**Save Version -> Save & Run All (Commit)**, not interactively. A browser tab
will not survive a multi-hour run; a committed version goes headless for 12 h.

### Flags that matter

- **`--cfg=paper256` is load-bearing.** The source net is named `paper256`; any
  other config changes layer shapes and `--resume` fails.
- `--mirror=1` doubles effective data for free -- Pokemon artwork has no
  meaningful chirality.
- `--aug=ada` is the entire reason for choosing this model over plain StyleGAN2.
- `--snap=10` writes a snapshot and a sample grid every 40 kimg, giving Phase 4
  about eight checkpoints. **The best snapshot is rarely the last one.**
- `--freezed` (FreezeD) stays at 0 for the pilot. It is the first knob to reach
  for if results disappoint, not a fixed value to guess at now.

### Why `--metrics=none` during training

KID belongs in Phase 4, where it ranks snapshots in one pass. Running it here
spends time inside the 12 h cap producing a number nobody acts on until then,
and `kid50k_full` generates 50k samples per evaluation.

The sample grids are the better in-flight signal anyway: the deliverable is a
game where people guess real from fake, so human judgement of the grids is
closer to the actual objective than KID is. KID's job in Phase 4 is to rank
snapshots that already look plausible.

### The resume caveat

`--resume` restores G, D, G_ema and the ADA augment pipeline, but **not the
optimizer state**. Across the 12 h cap that means Adam momentum resets at every
session boundary, so each class is sized to finish inside one session. Continuing
a class across sessions is supported (`RESUME` in step 1) but is second best.

---

## Expectations

Transfer learning at 100-shot scale is a demonstrated regime (Obama, Grumpy Cat,
Panda benchmarks). But those targets are *homogeneous* — one person, one cat.
These classes are not: Amphibian's 71 images span frogs and salamanders in
varied poses.

Expect quality to track class **coherence**, not just class size. Mammalian
(789) should produce genuinely good creatures; Amphibian is the one most likely
to disappoint.

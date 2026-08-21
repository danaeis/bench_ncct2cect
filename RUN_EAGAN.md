# Running Ea-GANs on our VinDr data (remote GPU server)

Fifth benchmark module. Same contract as the others: train on the shared
split, emit per-case synthetic CECT NIfTI (HU, source grid) + `manifest.csv`
for `../synthetic_CECT/benchmark.py`.

```bash
SPLIT=../synthetic_CECT/splits/split.json GPU=0 ./run_eagan.sh all
```

Two variants of the paper share this runner via `MODEL`:

```bash
SPLIT=../synthetic_CECT/splits/split.json GPU=0 MODEL=gea_gan ./run_eagan.sh all   # default
SPLIT=../synthetic_CECT/splits/split.json GPU=0 MODEL=dea_gan ./run_eagan.sh all
```

`gea_gan` puts the Sobel edge-similarity loss on the generator only; `dea_gan`
puts it on the discriminator too (an extra edge-map channel it also has to
judge as real/fake). Both are Ea-GANs, both worth having — they're cheap once
the adapter exists, and the paper's own ablation is exactly gEa-GAN vs dEa-GAN.

## Which Ea-GANs, and why this one is not upstream as-is

Upstream is [by-lab/Ea-GANs](https://github.com/by-lab/Ea-GANs) — the paper's
own repo. It is genuinely, thoroughly **3-D**: `Conv3d`/`BatchNorm3d`
throughout `networks.py`, a 3x3x3 Sobel kernel, and its own `AlignedDataset`
reads whole volumetric patches via SimpleITK, writing whole `.nii.gz` volumes
back out at test time. That is a different comparison than the one this
benchmark makes (every other model — ResViT, CyTran, CycleGAN, SwinUNETR,
TransUNet — trains and is scored on the same 2-D axial slices, same split, same
metrics), so `Ea-GANs/` here is a **2-D port**: `Conv2d`/`BatchNorm2d`, a
standard 3x3 Sobel operator (the 2-D restriction of upstream's kernel has no
exact form, so this is the literature-standard 2-D Sobel, not upstream's exact
numbers — both sides of the loss go through the identical filter, so it only
needs internal consistency, which it has), 4-D `(B,C,H,W)` tensors instead of
upstream's 5-D, and the same pix2pix AB-PNG slice loader ResViT already trains
on (`data/aligned_dataset.py`, copied verbatim from `ResViT/data/`).

The vendored copy this replaced was not upstream either — it had been hand-
modified in two incompatible, unfinished directions at once: `gea_gan_model.py`
prepended one-hot phase-condition channels to the generator input with no
dataset ever built to feed them, while `dea_gan_model.py` kept upstream's 5-D
volumetric tensors, and a shared edit to `networks.py`'s default `phase_channels`
silently broke `dea_gan_model.py`'s channel count too. Both also called
`.data[0]` on loss tensors (`Prerequisites.txt: torch==0.4.1`), which raises
`IndexError` on any current torch. None of that survived; this is a clean port
from upstream's actual `gea_gan_model.py`/`dea_gan_model.py`, not a repair of
the previous state. (It is still in git history if any of it is wanted back.)

**`--norm batch` is upstream's own default and is not overridden** — kept for
fidelity to the paper, unlike this benchmark's own UNet+PatchGAN ablations
(which default to instance norm; see `texture_consistency_findings.md` on why).
Override with `NORM=instance` if you want it on this benchmark's usual footing
instead — that changes what's being measured, so report which one ran.

## Stages

| stage | what it does |
|---|---|
| `setup` | verifies the vendored (2-D-ported) `Ea-GANs/` tree |
| `prep` | our NIfTI → pix2pix AB-PNG slices — reuses `ResViT/datasets/vindr` if already prepped, exactly like SwinUNETR/TransUNet (`RUN_BACKBONES.md`) |
| `train` | `--model gea_gan\|dea_gan`, A=NCCT → B=CECT |
| `test` | `netG` over the test slices → `<case>_<zzzz>_fake_B.png` |
| `reassemble` | slices → per-case CECT NIfTI (HU, source grid) + `manifest.csv` |

Tunables (env): `SPLIT`, `DATAROOT`, `SIZE` (256), `GPU`, `MODEL` (gea_gan/
dea_gan), `NAME`, `NITER`/`NITER_DECAY` (25/25), `BATCH` (4), `LR`, `NORM`,
`LAMBDA_A`, `LAMBDA_SOBEL`, `SAVE_EPOCH_FREQ`, `CKPT_EPOCH`, `RESUME`,
`EPOCH_COUNT`, `OUT_NIFTI`, `MIN_TISSUE_FRAC`.

`--which_model_netG unet_128` is upstream's own choice regardless of input
size (it names a downsampling *depth* — 7 halvings — not a required 128px
input; `define_G` has no deeper `unet_256`, so this is upstream's ceiling and
what every upstream training script uses). At `SIZE=256` the bottleneck is
2x2, not 1x1 — that's fine structurally, just worth knowing if you go looking
for it.

`--lambda_A 100` here, not upstream's own scripts' `300` — matches ResViT/
CyTran's L1 weight on this data rather than upstream's Coltea-tuned value.
Override with `LAMBDA_A=300` to run upstream's literal number instead; report
which one, since it changes the pixel/edge loss balance.

## Resuming — upstream never actually supported this

`train.py`'s epoch loop always started at 1, full stop; `--continue_train`
reloaded the weights but replayed the *entire* lr/Sobel-ramp schedule from
scratch on every restart — silently, since nothing errors. Added
`--epoch_count` (mirrors CycleGAN's own resume mechanism already in this repo)
so `RESUME=1` (the default) actually continues the schedule instead of just
the weights. Restart epoch comes from the newest **numbered** checkpoint
(`<epoch>_net_G.pth`, every `SAVE_EPOCH_FREQ`); `old_lr` itself is not
persisted, so the lr decay restarts from `--lr` on resume — safe (replays at
most `SAVE_EPOCH_FREQ` epochs' worth of schedule drift), not bit-identical to
an uninterrupted run. Pin `EPOCH_COUNT=N` to be exact, or `RESUME=0` to start
clean.

Smoke-test before committing GPU time:

```bash
./run_eagan.sh setup && ./run_eagan.sh prep
NITER=1 NITER_DECAY=0 SAVE_EPOCH_FREQ=1 ./run_eagan.sh train
./run_eagan.sh test && ./run_eagan.sh reassemble
```

## Details that took a pass through the source to get right

- **`--how_many` defaults to `inf`** in upstream's own `TestOptions` — unlike
  ResViT's `--how_many`/CycleGAN's `--num_test` (both default to 50 and
  silently truncate), so no truncation trap here. `run_eagan.sh` still passes
  `1000000` explicitly, matching the rest of this benchmark's belt-and-braces
  convention.
- **Upstream's own `test.py` does not force `batchSize=1`/`serial_batches`/
  `no_flip`** the way ResViT's and CycleGAN's do — and `tensor2im` only ever
  reads batch index 0. Left at upstream's train-oriented defaults, any
  `batchSize > 1` at test time would silently drop every sample after the
  first in each batch rather than error. `test.py` in this port forces all
  three itself, so `run_eagan.sh` doesn't need to remember to.
- **Output range.** `tensor2im` writes `[0,255]` display PNGs (copied from
  `ResViT/util/util.py`, already proven against `reassemble_nifti.py`), hence
  `--in_range 0_255` at reassembly — same as ResViT.
- **`test.py`'s visual is named `fake_B`**, so reassembly matches `_fake_B` —
  same suffix as ResViT, unlike CycleGAN's bare `_fake`.
- **Compat shim not needed.** Unlike CyTran, this port's `models/*.py` already
  import `ImagePool` correctly (`from util.image_pool import ImagePool`) — the
  broken `from util import ImagePool` pattern CyTran needed patched was never
  present here.

## Scoring

```bash
cd ../synthetic_CECT
python benchmark.py --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \
    --manifest gea_gan=../bench_ncct2cect/results/vindr_gea_gan_nifti/manifest.csv \
    --baseline ours --out analysis/bench_ncct2cect
# once dea_gan has also been run:
python benchmark.py --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \
    --manifest dea_gan=../bench_ncct2cect/results/vindr_dea_gan_nifti/manifest.csv \
    --baseline ours --out analysis/bench_ncct2cect
# → analysis/bench_ncct2cect/master_table.md
```

`analysis/bench_ncct2cect` — NOT `analysis/benchmark`, which is this repo's
*own* loss/architecture ablation store (see `RUN_ALL.md` §2).

# Running SwinUNETR and TransUNet on our VinDr data (remote GPU server)

Fifth and sixth benchmark modules. Same contract as every other model: train on
the shared split, emit per-case synthetic CECT NIfTI (HU, source grid) +
`manifest.csv` for `../synthetic_CECT/benchmark.py`.

```bash
SPLIT=../synthetic_CECT/splits/split.json GPU=0 ./run_swinunetr.sh all
SPLIT=../synthetic_CECT/splits/split.json GPU=1 ./run_transunet.sh all
```

Both wrappers are three lines over the shared [`run_backbone.sh`](run_backbone.sh);
they only pin `ARCH` and the default experiment name.

## Read this first: these two are not adapters

Every other model in the benchmark ships a **training pipeline**, so integrating
it is an I/O job — convert our NIfTI to its input format, run its `train.py`,
convert the outputs back. SwinUNETR and TransUNet ship an **architecture** plus a
*segmentation* trainer (Dice/CE on BTCV, Synapse, BraTS) that has nothing to do
with image synthesis. There is no `train.py` in either repo we could point at
NCCT→CECT.

So for these two we keep the upstream networks byte-for-byte and supply the
training loop ourselves, in three files at the repo root:

| file | role |
|---|---|
| [`backbone_common.py`](backbone_common.py) | dataset, generator factory, PatchGAN D, LSGAN loss, lr schedule |
| [`train_backbone.py`](train_backbone.py) | the training loop |
| [`infer_backbone.py`](infer_backbone.py) | test-set inference → `*_fake_B.npy` |

**The recipe is deliberately ResViT's**, so a gap in the final table is a
difference in architecture and not in optimisation:

```
loss_G = lambda_adv * LSGAN(D(A, G(A)), 1) + 100 * L1(G(A), B)
loss_D = 0.5 * lambda_adv * (LSGAN(D(A, G(A)), 0) + LSGAN(D(A, B), 1))
```

with a 3-layer conditional PatchGAN (`ndf=64`), `Adam(lr=2e-4, beta1=0.5)`, a
flat lr for `NITER` epochs then linear decay to zero over `NITER_DECAY`. Those
are exactly `ResViT/models/resvit_one.py` + `ResViT/options/*`.

`LAMBDA_ADV=0` drops the discriminator entirely and gives the pure supervised L1
regression that the SwinUNETR and TransUNet papers themselves use. That is a
worthwhile ablation — arguably the *fairer* setting for these two models — but
it is not the default, because everything else in the benchmark is adversarial.
Running both is cheap and makes a better table than picking one:

```bash
NAME=vindr_swinunetr_l1 LAMBDA_ADV=0 ./run_swinunetr.sh train
```

## Data: no new format, no second copy

Both read the **pix2pix AB-PNG slices** that `prep_benchmark_data.py --format
pix2pix` already writes for ResViT and CyTran. `DATAROOT` therefore defaults to
`ResViT/datasets/vindr`, and `prep` is a **no-op when that directory is already
populated** — a second copy of every axial slice is pure waste. Force a rebuild
with `REPREP=1`, or point `DATAROOT` elsewhere.

Tensors are `[-1, 1]` end to end (pix2pix convention), so reassembly uses
`--in_range neg1_1`.

No augmentation is applied: random flips would mirror left/right abdominal
anatomy, and random crops would break the 1:1 correspondence with the source
grid that reassembly depends on.

## Stages

| stage | what it does |
|---|---|
| `setup` | verifies the tree + deps; fetches the ImageNet-21k ViT init (TransUNet only) |
| `prep` | pix2pix AB-PNG slices, shared with ResViT (skipped if present) |
| `train` | `train_backbone.py` with the recipe above |
| `test` | `infer_backbone.py` over the test split → `<case>_<zzzz>_fake_B.npy` |
| `reassemble` | slices → per-case CECT NIfTI (HU, source grid) + `manifest.csv` |

Tunables (env): `SPLIT`, `DATAROOT`, `REPREP`, `SIZE` (256), `GPU`, `NAME`,
`CKPT_DIR`, `NITER` / `NITER_DECAY`, `BATCH`, `WORKERS`, `LR`, `LAMBDA_L1`,
`LAMBDA_ADV`, `SAVE_EPOCH_FREQ`, `FEATURE_SIZE`, `N_SLICES`, `WHICH_EPOCH`,
`CONTINUE`, `ALLOW_CPU`, `OUT_NIFTI`, `MIN_TISSUE_FRAC`.

## 2.5-D context (`N_SLICES`) — SwinUNETR only

```bash
N_SLICES=5  NAME=vindr_swinunetr_s5  ./run_swinunetr.sh all
N_SLICES=11 NAME=vindr_swinunetr_s11 ./run_swinunetr.sh all
```

`N_SLICES=N` (odd) feeds the N axial slices centred on z as input channels and
still predicts the single centre CECT slice. That is the same input geometry as
the `slices5_k2` / `slices11_k5` runs in `../synthetic_CECT`, so these numbers
land next to your own model's rather than only next to the other 2-D baselines.

- **No re-prep.** The stack is gathered at load time from the per-slice PNGs
  already on disk, so `N_SLICES` costs nothing but a little extra I/O.
- **Edges clamp, they don't pad.** At the top and bottom of a volume the index
  is clamped so the edge slice repeats — the model never sees fabricated air
  where anatomy should be.
- **Output geometry is unchanged** — one predicted slice per z — so inference
  and reassembly are identical for any N.
- The conditional discriminator widens to `N+1` input channels accordingly.

### Why not true 3-D at 5–11 slices

MONAI's 3-D SwinUNETR requires **every** spatial dim to be divisible by 32 (five
stages of patch-merging), so a 5- or 11-slice slab raises
`ValueError: spatial dimensions [2] ... must be divisible by 2**5`. The smallest
legal slab is 32. A depth-32 run is possible and would unlock the 3-D
self-supervised `model_swinvit.pt` weights, but it needs a volumetric data path
and sliding-window inference, and 62.2M params over a 32×256×256 sample will not
fit an 11 GB card. 2.5-D gets the z-context at 2-D cost.

### TransUNet

`N_SLICES` must stay 1. Its R50 stem is a fixed 3-channel ImageNet conv and
`forward()` only expands a *single* channel to 3, so an N-channel stack would
walk past that guard into a shape error. The runner refuses it with an
explanation rather than silently rebuilding a published layer.

## Never train on CPU by accident

`train`/`test` now preflight CUDA and **abort** if it is unavailable, printing
the torch build's CUDA version alongside the fix. This exists because a torch
built for a newer CUDA than the driver supports imports perfectly happily and
just reports `is_available() == False` — which cost a CycleGAN run a full day of
"Initialized with device cpu". Set `ALLOW_CPU=1` to override deliberately.

Smoke-test first:

```bash
./run_swinunetr.sh setup && ./run_swinunetr.sh prep
NITER=1 NITER_DECAY=0 SAVE_EPOCH_FREQ=1 ./run_swinunetr.sh train
./run_swinunetr.sh test && ./run_swinunetr.sh reassemble
```

## Model-specific notes

### SwinUNETR

- The network comes from the **`monai` package** (`monai.networks.nets.SwinUNETR`),
  not from vendored code. `SwinUNETR/` in this repo is MONAI's
  research-contributions reference (BTCV / BRATS21 / Pretrain) kept for
  provenance; its training scripts are 3-D segmentation and are not used.
- We run it with **`spatial_dims=2`** so it sees the same 256×256 axial slices as
  every other model in the benchmark.
- ⚠️ **It trains from scratch.** The published self-supervised SwinViT weights
  (`model_swinvit.pt`) are **3-D only** and cannot load into a `spatial_dims=2`
  model. TransUNet gets an ImageNet-21k initialisation and SwinUNETR does not —
  report that asymmetry next to its numbers rather than letting it read as an
  architecture result.
- `FEATURE_SIZE` (default 48) must be a multiple of 12. 25.1M params at 48.
- MONAI moved `img_size` from required → deprecated → removed across 1.3–1.6;
  `backbone_common.py` tries it and falls back, so any of those versions work
  (verified on 1.6.0).

### TransUNet

- Vendored at `TransUNet/` from
  [Beckschen/TransUNet](https://github.com/Beckschen/TransUNet), untouched.
- Config `R50-ViT-B_16`, `n_skip=3`, `patches.grid = (SIZE/16, SIZE/16)` — at
  SIZE=256 that is `(16, 16)`, which keeps the intended 16×16 patch after the
  ResNet stem's ÷16.
- `n_classes` is set to **1**, so the segmentation head emits one continuous map
  instead of class logits; a `tanh` bounds it to `[-1, 1]`. That is the smallest
  change that turns the segmenter into a generator, and it leaves every
  published layer intact.
- **1-channel input is the upstream-intended path**, not a hack:
  `vit_seg_modeling.py:387` already does `if x.size()[1] == 1: x = x.repeat(1,3,1,1)`
  because the R50 stem is a 3-channel ImageNet conv.
- It uses **the same `R50+ViT-B_16.npz`** that `run_resvit.sh` downloads, and
  `setup` reuses that file if present rather than re-fetching 400 MB. Set
  `VIT_INIT=0` to train from scratch instead — that is the like-for-like run
  against SwinUNETR, which has no 2-D pretrained weights available at all.
- 105.3M params — roughly 4× SwinUNETR. Drop `BATCH` to 2 if the GPU is small.

## Scoring

```bash
cd ../synthetic_CECT
python benchmark.py --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \
    --manifest swinunetr=../ncct2cect/results/vindr_swinunetr_nifti/manifest.csv \
    --manifest transunet=../ncct2cect/results/vindr_transunet_nifti/manifest.csv \
    --baseline ours --out analysis/benchmark
# → analysis/benchmark/master_table.md
```

## Deps

```bash
python -m venv .venv_backbone && source .venv_backbone/bin/activate
pip install -r requirements_backbone.txt
```

One env covers both models. Keep it separate from the ResViT / CyTran / SynDiff
envs (conflicting torch pins).

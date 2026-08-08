# Running ResViT on our VinDr data (remote GPU server)

ResViT is the first benchmark module (template for CyTran). Everything is driven
by [`run_resvit.sh`](run_resvit.sh); this file explains the prerequisites and the
corrected recipe.

## Prerequisites (must exist on the training server)

1. **A CUDA GPU** (ResViT needs CUDA ≥ 11.2). This repo's dev container has none
   — run there, not here.
2. **The VinDr NCCT/CECT NIfTI volumes**, reachable at the paths inside the split.
3. **The shared split** `../synthetic_CECT/splits/split.json` (seed 42), plus
   the sibling `synthetic_CECT` repo for scoring. `prep_benchmark_data.py` reads
   `hu_min/hu_max/target_phase` and the per-case `ncct`/`cect`/`seg` paths from it.
4. A dedicated Python env:
   ```bash
   python -m venv .venv_resvit && source .venv_resvit/bin/activate
   pip install -r requirements_resvit.txt   # pick the torch build matching your CUDA
   ```
   Keep it separate from the SynDiff / CFPS-Diff envs (conflicting torch pins).

## One command

```bash
SPLIT=../synthetic_CECT/splits/split.json GPU=0 ./run_resvit.sh all
```

Or run stages individually: `setup → prep → pretrain → finetune → test →
reassemble`. Tunables (env vars): `SPLIT`, `DATAROOT`, `SIZE` (256), `GPU`,
`NAME`, `PRE_NAME`, `NITER`/`NITER_DECAY`, `BATCH`, `OUT_NIFTI`.

Smoke-test first with tiny epochs before committing to a full run:
```bash
./run_resvit.sh setup && ./run_resvit.sh prep
NITER=1 NITER_DECAY=0 ./run_resvit.sh pretrain
NITER=1 NITER_DECAY=0 ./run_resvit.sh finetune
./run_resvit.sh test && ./run_resvit.sh reassemble
```

## What each stage does (and why it differs from BENCHMARK.md)

| stage | action |
|---|---|
| `setup` | verifies the vendored `ResViT/` tree, applies two compat shims, downloads the pretrained ViT checkpoint |
| `prep` | our NIfTI → pix2pix AB-PNG slices (`A`=NCCT, `B`=CECT) for train/val/test via `prep_benchmark_data.py` |
| `pretrain` | **stage 1** — pretrain the ART/`res_cnn` generator (no transformer) |
| `finetune` | **stage 2** — fine-tune the full `resvit` from the pretrained weights |
| `test` | inference on the test split → `<case>_<zzzz>_fake_B.png` |
| `reassemble` | slices → per-case CECT NIfTI (HU, source grid) + `manifest.csv` via `reassemble_nifti.py` |

- **Two-stage, not one.** ResViT/README recommends pretraining the convolutional
  (ART) blocks with `--which_model_netG res_cnn`, then fine-tuning `resvit` with
  `--pre_trained_transformer 1 --pre_trained_resnet 1 --pre_trained_path
  checkpoints/<pre>/latest_net_G.pth`. The `BENCHMARK.md` snippet showed only the
  single fine-tune command; the runner does both. One-to-one task →
  `--input_nc 1 --output_nc 1`, `AtoB` = NCCT→CECT.
- **Pretrained ViT checkpoint.** `models/transformer_configs.py` loads the exact
  path `ResViT/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz` (literal `+`,
  default `--vit_name Res-ViT-B_16`). `setup` fetches it from
  `https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz`. It is
  ~400 MB and is **not** committed.
- **Compatibility shims** (idempotent, re-applied at every stage that runs ResViT
  code; `ResViT/` is vendored so they are committed and normally a no-op):
  - `train.py`: `skimage.measure.compare_psnr` → `skimage.metrics.
    peak_signal_noise_ratio` (renamed/removed in scikit-image ≥ 0.18).
  - `util/visualizer.py`: the top-level `from scipy.misc import imresize` (removed
    in SciPy ≥ 1.3) is replaced with a no-op; `imresize` is only reached when
    `aspect_ratio != 1.0`, which never happens in our path.
- **Output range.** `util.tensor2im` writes display PNGs in `[0,255]`, so
  reassembly uses `--in_range 0_255` and `--slice_suffix _fake_B`. Testing passes
  `--display_id 0` (no visdom) and `--how_many 1000000` so the whole test set is
  written (the default 50 would truncate).

## Scoring

```bash
cd ../synthetic_CECT
python benchmark.py --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \
    --manifest resvit=../ncct2cect/ResViT/results/vindr_nifti/manifest.csv \
    --baseline ours --out analysis/benchmark
# → analysis/benchmark/master_table.md
```

## Next modules (same adapter pattern)

- **CyTran** — same pix2pix slices, but *not* a near-clone: the repo needs import
  shims, its own dataset module and a hand-written inference script. Done —
  `run_cytran.sh` / [RUN_CYTRAN.md](RUN_CYTRAN.md).
- **SynDiff** — `prep_benchmark_data.py --format mat`; reassemble `--in_range neg1_1`.
- **CFPS-Diff** — reads NIfTI natively; roughest, do last; multiphase output, keep
  only the venous volume.

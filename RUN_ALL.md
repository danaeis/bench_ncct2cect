# Execution order and one growing results table

Every model here trains independently and is scored the moment it finishes. The
table grows as they land — nothing is recomputed, and no model waits for another.

---

## 0. Before anything: check the GPU you intend to use

```bash
CUDA_VISIBLE_DEVICES=0 python preflight_gpu.py
CUDA_VISIBLE_DEVICES=1 python preflight_gpu.py
```

Five seconds, and it catches the three failures that otherwise surface *after*
the model is built and the checkpoints are loaded:

1. `is_available()` False — torch built for a newer CUDA than the driver serves.
2. the card's `sm_XX` missing from `torch.cuda.get_arch_list()`.
3. cuDNN unable to run a convolution on this architecture.

Every runner now calls this before `train`/`test` and refuses to start if it
fails (`ALLOW_CPU=1` overrides). This is why `CUDNN_STATUS_NOT_INITIALIZED` on
the 1080 Ti now stops the run in five seconds instead of after model init.

### What the preflight found on this box

**GPU 1 (GTX 1080 Ti, `sm_61`) is permanently unusable with this environment.**
torch says so itself:

```
RuntimeError: cuDNN version 92000 is not compatible with devices with SM < 7.5
```

cuDNN 9.x requires SM ≥ 7.5 (Turing or later); the 1080 Ti is Pascal. Nothing
short of downgrading to a much older torch bundling cuDNN 8.x will change that,
and that would break the rest of the stack. **Treat this as a one-GPU machine
and run everything serially on GPU 0.**

**GPU 0 (RTX 3090 Ti, `sm_86`) is a supported card with a broken cuDNN runtime.**
`sm_86` is well above the cutoff and 13.7 GiB were free, yet a convolution still
raised `CUDNN_STATUS_NOT_INITIALIZED`. That combination means the *library*, not
the hardware: torch reports the cuDNN it was **built** against (9.2.0), but
`dlopen`s `libcudnn` from the filesystem at **run** time, and something else is
winning — usually a stale system cuDNN on `LD_LIBRARY_PATH`, or a broken
`nvidia-cudnn-cu12` wheel in the conda env.

Diagnose which library is actually loaded:

```bash
echo "$LD_LIBRARY_PATH"
pip list | grep -i nvidia-cudnn
python -c "import torch;torch.randn(1,device='cuda');import os;os.system(f'grep -i cudnn /proc/{os.getpid()}/maps')"
```

Fixes, in order of likelihood:

```bash
unset LD_LIBRARY_PATH                              # stale system cuDNN shadowing the wheel
pip install --force-reinstall nvidia-cudnn-cu12    # broken/partial wheel
# or build a clean env from requirements_backbone.txt / requirements_cyclegan.txt
```

### Keep training while you fix it: `DISABLE_CUDNN=1`

```bash
DISABLE_CUDNN=1 GPU=0 ./run_cyclegan.sh train
```

Convolutions fall back to native CUDA kernels — typically ~1.5–3× slower for
these models, but it needs no reinstall and unblocks every runner. It works for
the vendored repos too: the runners route their entrypoints through
[`nocudnn.py`](nocudnn.py), which flips `torch.backends.cudnn.enabled` off before
the first conv while preserving `argv` and `__main__`, so upstream code is never
patched for a local environment problem.

This cannot rescue GPU 1 — there the failure is inside cuDNN's own init and torch
refuses the device outright, before any of our code runs.

---

## 1. Models to run

`SPLIT=../synthetic_CECT/splits/split.json` throughout. All but CycleGAN share
one prep of the pix2pix slices (CycleGAN needs its own `{train,val,test}{A,B}/`
layout).

| # | model | command | notes |
|---|---|---|---|
| 1 | ResViT | `GPU=0 ./run_resvit.sh all` | two-stage; resumable (`RESUME=1` default) |
| 2 | CyTran | `GPU=0 ./run_cytran.sh all` | reuses ResViT's slices |
| 3 | CycleGAN (2-D) | `GPU=0 ./run_cyclegan.sh all` | unpaired control; own `--format cyclegan` prep |
| 4 | SwinUNETR (2-D) | `GPU=0 ./run_swinunetr.sh all` | `N_SLICES=1` |
| 5 | SwinUNETR (2.5-D, 5) | `GPU=0 N_SLICES=5 NAME=vindr_swinunetr_s5 ./run_swinunetr.sh all` | 5-slice z-context |
| 6 | SwinUNETR (2.5-D, 11) | `GPU=0 N_SLICES=11 NAME=vindr_swinunetr_s11 ./run_swinunetr.sh all` | 11-slice z-context |
| 7 | TransUNet | `GPU=0 ./run_transunet.sh all` | ImageNet-21k init; `N_SLICES` must stay 1 |
| 8 | gEa-GAN | `GPU=0 ./run_eagan.sh all` | reuses ResViT's slices; 2-D port of by-lab/Ea-GANs (see `RUN_EAGAN.md`) |
| 9 | dEa-GAN | `GPU=0 MODEL=dea_gan ./run_eagan.sh all` | same runner, edge loss on D too |

**SynDiff dropped** (see `BENCHMARK.md`'s notes) — measured ~2s/it steady state
at batch=1, projecting multi-week wall-clock; not worth the compute budget for
this thesis. `run_syndiff.sh`/`RUN_SYNDIFF.md` still work if revisited later.

Optional ablations, cheap and worth having:

```bash
LAMBDA_ADV=0 NAME=vindr_swinunetr_l1 ./run_swinunetr.sh all   # pure-L1, no discriminator
VIT_INIT=0   NAME=vindr_transunet_scratch ./run_transunet.sh all  # like-for-like vs SwinUNETR
```

Run them **one at a time on GPU 0** unless you have checked free VRAM — another
user already holds ~7.9 GB there. `nvidia-smi` before each launch.

Rough sizing at 25,251 train slices/epoch: SwinUNETR 25.1M params, TransUNet
105.3M, CycleGAN 4 nets, SynDiff 8 nets. Set `MIN_TISSUE_FRAC=0.05` on any run
you want shorter — it drops near-empty slices from **train/val only**, never
test, so reconstruction stays complete.

---

## 2. Scoring — one table that grows

`benchmark.py` already has an accumulating store, so you do **not** re-score
finished models. Each run caches its per-case rows under `<out>/store/` and the
table is rebuilt from everything cached so far.

Use a directory of your own so your existing `analysis/benchmark` is untouched:

```bash
cd ../synthetic_CECT

python benchmark.py \
    --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \
    --out analysis/bench_ncct2cect \
    --baseline ours \
    --manifest resvit=../bench_ncct2cect/ResViT/results/vindr_nifti/manifest.csv
```

Then, as each further model finishes, add only the new one:

```bash
python benchmark.py --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \
    --out analysis/bench_ncct2cect --baseline ours \
    --manifest cyclegan=../bench_ncct2cect/CycleGAN/results/vindr_nifti/manifest.csv

python benchmark.py --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \
    --out analysis/bench_ncct2cect --baseline ours \
    --manifest swinunetr_s5=../bench_ncct2cect/results/vindr_swinunetr_s5_nifti/manifest.csv
```

Each call rewrites `analysis/bench_ncct2cect/master_table.{md,csv}` and
`per_case.csv` containing **that model plus every previously scored one**.

Manifest paths per model:

| model | manifest |
|---|---|
| ResViT | `../bench_ncct2cect/ResViT/results/vindr_nifti/manifest.csv` |
| CyTran | `../bench_ncct2cect/CyTran/results/vindr_nifti/manifest.csv` |
| CycleGAN | `../bench_ncct2cect/CycleGAN/results/vindr_nifti/manifest.csv` |
| SwinUNETR | `../bench_ncct2cect/results/vindr_swinunetr_nifti/manifest.csv` |
| SwinUNETR s5 / s11 | `../bench_ncct2cect/results/vindr_swinunetr_s5_nifti/manifest.csv` (…`_s11_`) |
| TransUNet | `../bench_ncct2cect/results/vindr_transunet_nifti/manifest.csv` |
| gEa-GAN | `../bench_ncct2cect/results/vindr_gea_gan_nifti/manifest.csv` |
| dEa-GAN | `../bench_ncct2cect/results/vindr_dea_gan_nifti/manifest.csv` |
| SynDiff (dropped) | `../bench_ncct2cect/SynDiff/results/vindr_nifti/manifest.csv` — not run, see `BENCHMARK.md` |

Housekeeping:

```bash
python benchmark.py --weights … --out analysis/bench_ncct2cect --list_store
python benchmark.py --weights … --out analysis/bench_ncct2cect --drop swinunetr_s5
python benchmark.py --weights … --out analysis/bench_ncct2cect --fresh --manifest …
```

`--list_store` shows what is cached, `--drop` retires a superseded run, `--fresh`
builds the table from only this invocation while still updating the store.

### The invariants

The store fingerprints four settings (`_fingerprint` in `benchmark.py`):
`--weights`, `--hu_min`, `--hu_max`, `--gen_not_hu`. **Keep those identical on
every call.** Change one and the older entries are reported stale and dropped
from the table rather than silently pooled — which is the right behaviour, since
a model scored on a different HU window is not the same quantity.

Two things worth knowing:

- **`--perceptual` is deliberately *not* fingerprinted.** You can add LPIPS/FID
  later; models scored without it carry NaN for those columns and still pool
  correctly. So there is no need to decide up front.
- **`--min_body_frac` is *not* fingerprinted either** — but it does change the
  numbers. Nothing will warn you, so keep it at its default (or the same value)
  across every run yourself.

To include your own model for comparison, score it into the same store once:

```bash
python benchmark.py --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \
    --out analysis/bench_ncct2cect --baseline ours \
    --manifest ours=../out_synthesis_train/literature_baseline_l1_organ_curriculum/phase_infer/manifest.csv
```

`--baseline ours` then gives paired significance tests against it in every
subsequent table.

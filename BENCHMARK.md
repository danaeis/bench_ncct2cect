# NCCT→CECT benchmark — models & integration

This repo holds the **competing models** for the NCCT→CECT synthesis benchmark.
Scoring lives in the sibling repo `../synthetic_CECT` (`benchmark.py` +
`metrics.py` + `orgFeatXGB_CTPhase/phase_eval.py`).

**Integration contract:** every model, however it trains, must emit synthetic CECT
volumes as **NIfTI on each test case's source grid** (same shape as the real CECT,
values in **HU**) and a manifest CSV with columns
`gen_path,real_path,mask_path,target_phase`. `benchmark.py` then scores all models
identically. Nothing about a model's internals matters to the harness — only that
it produces those volumes for the shared test cases.

**Fairness:** all models train and evaluate on the *same* case-level split
(`../synthetic_CECT/splits/split.json`, seed 42) and are scored on the same
test cases, masks, and HU window. Retraining these repos on our data at our scale
does **not** reproduce their papers' headline numbers — this is a controlled
same-data comparison, not a reproduction.

---

## Models

### Pre-existing (GAN / CNN family)
- `Ea-GANs/` — edge-aware GAN (cross-modality synthesis). Integrated as
  `run_eagan.sh` (see `RUN_EAGAN.md`) — a **2-D port** of upstream
  [by-lab/Ea-GANs](https://github.com/by-lab/Ea-GANs), which is natively
  volumetric (Conv3d, whole-3D-patch SimpleITK I/O). Ported to Conv2d + this
  benchmark's pix2pix AB-PNG slices so it is comparable to ResViT/CyTran/
  CycleGAN on the same split and metrics; see `RUN_EAGAN.md` for exactly what
  changed. Both paper variants run via `MODEL=gea_gan|dea_gan`.
- `VCE_CESM/` — virtual contrast enhancement.
- `pix2pix3D-CT/` — 3D pix2pix for CT.
- ⚠️ `Ea_GANs/` (no hyphen) — a broken duplicate zip artifact, removed; `Ea-GANs/`
  (hyphen) above is the real one, now integrated.
- ⚠️ `UnpairedImageTranslation/` — **superseded, do not use.** It is Kaji's
  **Chainer** reimplementation of CycleGAN; Chainer has been unmaintained since
  2019 and it would need a CUDA-10.2 `cupy` wheel (both sit unused in
  `vindr_ds/`). `CycleGAN/` below is the PyTorch original and covers the same
  model on a maintained stack.

### Added for SOTA + diffusion coverage (vendored into this repo)

| dir | model | type | paper | entrypoints | input format |
|---|---|---|---|---|---|
| `SynDiff/` | SynDiff | adversarial **diffusion** | TMI'23, [icon-lab](https://github.com/icon-lab/SynDiff) | `train.py`, `test.py` | `.mat`, `(#img,W,H)`, values 0–1 |
| `CFPS-Diff/` | CFPS-Diff | conditional **diffusion** (paper: NCCT→multiphase CECT) | MICCAI'25, [Kindyz](https://github.com/Kindyz/CFPS-Diff) | `main/train_CBSI_gen.py`, `main/Inference_CBSI.py` | brain-MR NIfTI dirs (`T1/T2F/T1C/ROI/Brain_mask.nii.gz`) — ⚠️ not CT, see below |
| `ResViT/` | ResViT | transformer-GAN | TMI'22, [icon-lab](https://github.com/icon-lab/ResViT) | `train.py`, `test.py` | pix2pix-style aligned pairs |
| `CyTran/` | CyTran | cycle **transformer** | Neurocomputing'23, [ristea](https://github.com/ristea/cycle-transformer) | `train.py`, `test.py`, `options/base_options.py` (data path) | CycleGAN/pix2pix-style; ships Coltea-Lung-CT-100W |
| `CycleGAN/` | CycleGAN | **unpaired** GAN | ICCV'17, [junyanz](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) | `train.py`, `test.py` | two dirs per phase (`trainA`/`trainB`) |
| `TransUNet/` | TransUNet | hybrid R50+ViT **U-Net** | arXiv'21, [Beckschen](https://github.com/Beckschen/TransUNet) | *architecture only* — see below | pix2pix AB-PNG (via our trainer) |
| — (`monai`) | SwinUNETR | Swin-transformer **U-Net** | CVPR'22 / MICCAI'21, [MONAI](https://github.com/Project-MONAI/research-contributions) | *architecture only* — see below | pix2pix AB-PNG (via our trainer) |

Notes:
- **ResViT and CyTran share the pix2pix/CycleGAN codebase** (options/, aligned-pair
  loaders) — one adapter pattern covers both. CyTran's README states it "is similar
  with CycleGAN-and-pix2pix".
- ⚠️ **SynDiff dropped from the active run (compute budget).** Measured ~2.1–2.4s/it
  steady state at `BATCH=1` with the default `NUM_EPOCH=50` (25,251
  iterations/epoch, no `MIN_TISSUE_FRAC` filtering) — multi-week wall-clock on
  this box, not worth it for this thesis's timeline. The adapter (`run_syndiff.sh`,
  `RUN_SYNDIFF.md`) is not broken; revisit with a larger `BATCH` (VRAM headroom
  was there — 17/80GB used), a shorter `NUM_EPOCH`, or a free multi-GPU window
  if SynDiff's numbers turn out to matter later.
- **SynDiff wants `.mat`** inputs (`(#images, W, H)`, [0,1]) — its adapter must
  convert NCCT/CECT slices to `.mat` and convert generated slices back to HU NIfTI.
- **CycleGAN is the unpaired control.** Our slices are paired on disk, but
  `--dataset_mode unaligned` draws A and B independently, so it is trained purely
  by cycle-consistency. That is deliberate: it measures what the unpaired
  objective costs on data where pairing *is* available. Switching it to aligned
  mode would just make it pix2pix, which ResViT already covers.
- ⚠️ **SwinUNETR and TransUNet are architectures, not pipelines.** Both ship a
  *segmentation* trainer (Dice/CE on BTCV/Synapse/BraTS) with no synthesis path,
  so unlike every other entry there is no upstream `train.py` to point at
  NCCT→CECT. We keep their networks byte-for-byte and supply the training loop
  ourselves — `backbone_common.py` / `train_backbone.py` / `infer_backbone.py` —
  using **ResViT's exact objective** (conditional PatchGAN + 100·L1, LSGAN,
  Adam 2e-4/β₁=0.5, flat-then-linear-decay lr) so the comparison isolates
  architecture rather than optimisation. `LAMBDA_ADV=0` gives the pure-L1
  ablation those two papers actually use. They read ResViT's existing pix2pix
  slices, so there is no third data format and no second copy on disk.
- ⚠️ **CFPS-Diff's published code does not implement the paper it advertises.** The
  README describes NCCT→multiphase CECT, but every uploaded script is a *brain-MRI*
  contrast-enhancement model (BraTS): T1 + T2-FLAIR → T1Gd, plus an enhancing-tumor
  classifier. Verified against upstream HEAD `674c96a` — our vendored copy is
  byte-identical, so this is upstream's state, not a bad clone. Details below.
- Deps per repo stay inside each subdir; do not merge into a global env (SynDiff
  torch>=1.7 vs CFPS-Diff torch==2.0 conflict).

### Watch-list — no usable public code yet (add when released)
- **PHASOR** (arXiv'26) — volumetric video-diffusion, "code on acceptance". The
  current SOTA frontier; strongest candidate to add next.
- **SC-BBDM** (Brownian-bridge CTA, arXiv'25) — no code link.
- **MAN-GAN** (SciRep'25), **IEEE 10838281**, **MLMI'25 §14**, **Acad.Radiology
  '24** — no public code; would require reimplementation (out of scope).

---

## Running on OUR VinDr NIfTI data — the concrete path

Two shared tools convert our NIfTI ↔ the repos' slice formats, so you never touch
their internals:

- **`prep_benchmark_data.py`** — split.json → 2-D axial slices (pix2pix AB-PNG for
  ResViT/CyTran, or `.mat` for SynDiff). Windows HU→[0,1], resizes to 256, writes
  `slice_index.csv`. Round-trip verified at ~1.2 HU error on CT-like data.
- **`reassemble_nifti.py`** — a model's per-slice outputs → per-case CECT NIfTI on
  the real grid + a scoring `manifest.csv`.

### ResViT (do first — template for CyTran)
```bash
# 1. our NIfTI → pix2pix AB-PNG slices on the shared split
python prep_benchmark_data.py --split ../synthetic_CECT/splits/split.json \
    --format pix2pix --out ResViT/datasets/vindr --size 256

# 2. train  (single-channel CT; AtoB = NCCT→CECT). See ResViT/README for the
#    two-stage pretrain→resvit recipe; minimal:
cd ResViT
python3 train.py --dataroot datasets/vindr --name vindr_resvit --model resvit_one \
    --which_model_netG resvit --which_direction AtoB --lambda_A 100 \
    --dataset_mode aligned --norm batch --input_nc 1 --output_nc 1 \
    --loadSize 256 --fineSize 256 --niter 25 --niter_decay 25 --gpu_ids 0

# 3. inference on the test split → per-slice images
python3 test.py --dataroot datasets/vindr --name vindr_resvit --model resvit_one \
    --which_model_netG resvit --which_direction AtoB --dataset_mode aligned \
    --norm batch --input_nc 1 --output_nc 1 --loadSize 256 --fineSize 256 --phase test

# 4. slices → NIfTI + manifest  (ResViT saves *_fake_B.png; range [-1,1]→ use neg1_1
#    if it saves raw, or 0_255 if it saves display PNGs — check one file)
cd ..
python reassemble_nifti.py --index ResViT/datasets/vindr/slice_index.csv \
    --slices_dir ResViT/results/vindr_resvit/test_latest/images \
    --slice_suffix _fake_B --in_range 0_255 --out ResViT/results/vindr_nifti
```

### CyTran
Same pix2pix slices (`--format pix2pix`); CyTran's `data/aligned_dataset.py` reads
the identical layout. Train per CyTran/README pointing `--dataroot` at
`CyTran/datasets/vindr` (or reuse ResViT's output dir), then reassemble the same way
(check its generated-slice suffix).

### CycleGAN
Fully driven by [`run_cyclegan.sh`](run_cyclegan.sh) — see [RUN_CYCLEGAN.md](RUN_CYCLEGAN.md).
```bash
SPLIT=../synthetic_CECT/splits/split.json GPU=0 ./run_cyclegan.sh all
```
Needs its own prep layout (`--format cyclegan` → `{train,val,test}{A,B}/`). Three
things to know: this HEAD has **no `--gpu_ids`** (device comes from
`CUDA_VISIBLE_DEVICES` via `util.init_ddp()`); `--num_test` defaults to 50 and
would truncate the test set; and inference runs `--model test --model_suffix _A`,
whose visual is named `fake`, so reassembly matches `_fake` — **not** `_fake_B`.

### SwinUNETR and TransUNet
Fully driven by [`run_swinunetr.sh`](run_swinunetr.sh) / [`run_transunet.sh`](run_transunet.sh),
thin wrappers over the shared [`run_backbone.sh`](run_backbone.sh) — see
[RUN_BACKBONES.md](RUN_BACKBONES.md).
```bash
SPLIT=../synthetic_CECT/splits/split.json GPU=0 ./run_swinunetr.sh all
SPLIT=../synthetic_CECT/splits/split.json GPU=1 ./run_transunet.sh all
```
`DATAROOT` defaults to ResViT's prepped slices and `prep` is a no-op when they
already exist. Two asymmetries to report with the numbers: TransUNet is
initialised from the **same** `R50+ViT-B_16.npz` ResViT downloads (`VIT_INIT=0`
to disable), while **SwinUNETR trains from scratch** — MONAI's self-supervised
SwinViT weights are 3-D only and cannot load into the `spatial_dims=2` model we
run to keep it comparable with the rest of the 2-D benchmark.

### SynDiff
Fully driven by [`run_syndiff.sh`](run_syndiff.sh) — see [RUN_SYNDIFF.md](RUN_SYNDIFF.md).
```bash
SPLIT=../synthetic_CECT/splits/split.json GPU=0 ./run_syndiff.sh all
```
Three things to know: it JIT-compiles CUDA kernels at import (needs nvcc + ninja +
pythonX.Y-dev — `setup` preflights this); `--size` must be exactly 256, because
`LoadDataSet` pads to a hardcoded 256 and any other size shifts the anatomy; and its
`test.py` cannot be used, so inference runs through [`infer_syndiff.py`](infer_syndiff.py),
which reuses upstream's own diffusion sampler and only replaces the I/O.
`LoadDataSet` also transposes every slice (MATLAB column-major convention);
`infer_syndiff.py` transposes back so reassembly stays aligned.

### CFPS-Diff (blocked — published code is the wrong modality)
Do **not** budget this as an adapter job. The repo is vendored and current with
upstream, but the code is a brain-MRI model, not a CT one:

- `main/Dataset_gen.py` walks `ET-1/` and `ET-0/` class dirs and loads
  `T1.nii.gz`, `T2F.nii.gz`, `T1C.nii.gz`, `ROI.nii.gz`, `Brain_mask.nii.gz`
  per case — no NCCT/CECT path anywhere.
- `main/train_CBSI_gen.py` — `--cc 2` is documented as "condition_channels:
  non-contrast MR (T1 and T2-FLAIR)", `--class_cond 2` is "ET label types
  (enhancing and non-enhancing)", `--output_nc 1` is "output T1Gd image".
  The training loop comments `x` as T1Gd and `x_cond` as T1+T2-FLAIR.
- `main/Inference_CBSI.py` hardcodes `./glioma_data/MICCAI_2023/BraTS-GLI/` and
  `BraTS-Africa/`, and classifies enhancement rather than emitting phases.
- Intensities are MR-harmonized to `--MR_min 0 / --MR_max 255`, `--ImageSize 424`;
  there is no HU windowing.
- The only CT trace in the whole repo is one stale docstring
  (`main/Networks_DDPM_trainer.py:382`, "Conditional input (e.g., NCCT)").
- README-referenced files that don't exist upstream: `main/train_CFPS-Diff_gen.py`,
  `main/Inference.py`, `Error_troubleshooting.txt`.

The diffusion backbone (`Networks_UNet_DDPM.py`, `Networks_DDPM_trainer.py`,
`Networks_model_diffusion.py`) is generic and reusable, so a port is *possible*:
2-ch MR condition → 1-ch NCCT, `class_cond` ET labels → phase labels, drop the
segmentation aux loss (no CT analog for ROI/Brain_mask), swap harmonization for
HU windowing. That is a reimplementation, not an adapter — scope it deliberately
or move CFPS-Diff to the watch-list until the authors upload the CT code.

### Fallback adapter contract (any repo not covered above)
Read `../synthetic_CECT/splits/split.json` → convert → train → infer → export
CECT **NIfTI on the source grid (HU)** + manifest CSV. Reuse
`../synthetic_CECT/infer_volume.py` stitching for patch/slice outputs.

## Scoring (after any model produces a manifest)

Results **accumulate**: each model's per-case rows are cached under
`<out>/store/`, so score a model once, when it finishes, and it stays in every
table printed afterwards. Every cross-model quantity (best/second marks, paired
t-tests, level-recovery regressions) is recomputed from the merged rows on each
run, so the 6th model retroactively updates the table for the first 5.

```bash
cd ../synthetic_CECT
W=orgFeatXGB_CTPhase/xgb_vindr_full.pkl
OUT=analysis/bench_ncct2cect   # NOT analysis/benchmark — that is this repo's OWN
                                # loss/architecture ablation store (RUN_ALL.md §2);
                                # scoring externals into it mixes two different
                                # comparisons into one table.

# as each model finishes — one invocation per model, nothing is re-scored
python benchmark.py --weights $W --out $OUT --perceptual --baseline ours \
    --manifest ours=../out_synthesis_train/literature_baseline_l1_organ_curriculum/phase_infer/manifest.csv
python benchmark.py --weights $W --out $OUT --perceptual --baseline ours \
    --manifest resvit=../bench_ncct2cect/ResViT/results/vindr_nifti/manifest.csv
python benchmark.py --weights $W --out $OUT --perceptual --baseline ours \
    --manifest syndiff=../bench_ncct2cect/SynDiff/results/vindr_nifti/manifest.csv
# → analysis/bench_ncct2cect/master_table.md, with every model scored so far

python benchmark.py --weights $W --out $OUT --list_store   # what is cached
python benchmark.py --weights $W --out $OUT --drop resvit  # retire a superseded run
```

Pass `--perceptual` on the run that scores each model (it needs `lpips` +
`pytorch-fid`): LPIPS and FID are cached with that model's rows, so the
perceptual tables render on later runs whether or not those runs use the flag.
FID is per-model distributional and is computed against the real slices of that
model's own cases, which is what makes caching it valid.

Entries scored under a different HU window, `--gen_not_hu`, or a different phase
classifier are **excluded and reported**, never pooled — those columns would not
be the same quantity. Re-score them, or use a separate `--out`.

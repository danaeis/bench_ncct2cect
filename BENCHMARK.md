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
- `Ea-GANs/` — edge-aware GAN (cross-modality synthesis).
- `UnpairedImageTranslation/` — CycleGAN-family unpaired translation.
- `VCE_CESM/` — virtual contrast enhancement.
- `pix2pix3D-CT/` — 3D pix2pix for CT.
- ⚠️ `Ea_GANs/` — appears to be a broken nested clone (contains only
  `Ea_GANs/Ea_GANs`); `Ea-GANs/` (hyphen) is the real one. Recommend removing.

### Added for SOTA + diffusion coverage (vendored into this repo)

| dir | model | type | paper | entrypoints | input format |
|---|---|---|---|---|---|
| `SynDiff/` | SynDiff | adversarial **diffusion** | TMI'23, [icon-lab](https://github.com/icon-lab/SynDiff) | `train.py`, `test.py` | `.mat`, `(#img,W,H)`, values 0–1 |
| `CFPS-Diff/` | CFPS-Diff | conditional **diffusion** (NCCT→multiphase CECT) | MICCAI'25, [Kindyz](https://github.com/Kindyz/CFPS-Diff) | `main/train_CFPS-Diff_gen.py`, `main/Inference.py` | see `main/`; ships `Error_troubleshooting.txt` |
| `ResViT/` | ResViT | transformer-GAN | TMI'22, [icon-lab](https://github.com/icon-lab/ResViT) | `train.py`, `test.py` | pix2pix-style aligned pairs |
| `CyTran/` | CyTran | cycle **transformer** | Neurocomputing'23, [ristea](https://github.com/ristea/cycle-transformer) | `train.py`, `test.py`, `options/base_options.py` (data path) | CycleGAN/pix2pix-style; ships Coltea-Lung-CT-100W |

Notes:
- **ResViT and CyTran share the pix2pix/CycleGAN codebase** (options/, aligned-pair
  loaders) — one adapter pattern covers both. CyTran's README states it "is similar
  with CycleGAN-and-pix2pix".
- **SynDiff wants `.mat`** inputs (`(#images, W, H)`, [0,1]) — its adapter must
  convert NCCT/CECT slices to `.mat` and convert generated slices back to HU NIfTI.
- **CFPS-Diff** is the roughest repo (Python 3.8, torch 2.0+cu118, troubleshooting
  file). Do it last. It targets multiphase output — we score only the venous phase.
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

### CFPS-Diff (last — roughest, diffusion, multiphase)
Reads NIfTI natively via `main/Nii_utils.py` / `main/Dataset_gen.py`; point its
dataset config at our volumes directly (no prep tool needed). It emits multiphase
output — keep only the venous volume, then build a manifest by hand or with a thin
wrapper, and score. Consult `CFPS-Diff/Error_troubleshooting.txt` first.

### Fallback adapter contract (any repo not covered above)
Read `../synthetic_CECT/splits/split.json` → convert → train → infer → export
CECT **NIfTI on the source grid (HU)** + manifest CSV. Reuse
`../synthetic_CECT/infer_volume.py` stitching for patch/slice outputs.

## Scoring (after any model produces a manifest)

```bash
cd ../synthetic_CECT
python benchmark.py --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \
    --manifest ours=../out_synthesis_train/literature_baseline_l1_organ_curriculum/phase_infer/manifest.csv \
    --manifest resvit=../ncct2cect/ResViT/<out>/manifest.csv \
    --baseline ours --out analysis/benchmark
# → analysis/benchmark/master_table.md
```

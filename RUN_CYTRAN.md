# Running CyTran on our VinDr data (remote GPU server)

CyTran is benchmark module **2 of 4** (after ResViT, before SynDiff and
CFPS-Diff). Everything is driven by [`run_cytran.sh`](run_cytran.sh); this file
explains the prerequisites, what had to be adapted, and how to score the result.

## One command

```bash
SPLIT=../synthetic_CECT/splits/split.json GPU=0 ./run_cytran.sh all
```

Stages run individually as `setup → prep → train → infer → reassemble`.

Smoke-test with tiny epochs before committing GPU time — note `SAVE_EPOCH_FREQ=1`,
without it a 1-epoch run finishes with **no checkpoint on disk**:

```bash
./run_cytran.sh setup && ./run_cytran.sh prep
NITER=1 NITER_DECAY=0 SAVE_EPOCH_FREQ=1 ./run_cytran.sh train
./run_cytran.sh infer && ./run_cytran.sh reassemble
```

## Prerequisites

Same as ResViT: a CUDA GPU, the VinDr NIfTI volumes at the paths inside the
split, the shared split `../synthetic_CECT/splits/split.json` (seed 42), and a
dedicated env:

```bash
python -m venv .venv_cytran && source .venv_cytran/bin/activate
pip install -r requirements_cytran.txt      # torch build matching your CUDA
```

CyTran's README asks for torch 1.7.x and anything ≥1.7 works, so this env **can
be shared with ResViT** if you prefer one less venv. Keep it away from the
SynDiff / CFPS-Diff envs — those torch pins conflict.

## Tunables (env vars)

| var | default | note |
|---|---|---|
| `SPLIT` | `../synthetic_CECT/splits/split.json` | shared 97/20/20 case split |
| `DATAROOT` | `CyTran/datasets/vindr` | prepped slices |
| `SIZE` | `256` | must be divisible by `2^N_DOWNSAMPLING` |
| `GPU` | `0` | `-1` = CPU |
| `NAME` | `vindr_cytran` | checkpoint/results dir |
| `NITER` / `NITER_DECAY` | `25` / `25` | `--n_epochs` / `--n_epochs_decay` |
| `BATCH` | `2` | CyTran's default |
| `LR` | `0.0001` | CyTran's default (ResViT used 1e-3 for fine-tune) |
| `SAVE_EPOCH_FREQ` | `5` | **total epochs must be a multiple of this** |
| `CKPT_EPOCH` | `latest` | which checkpoint `infer` loads |
| `NGF_CYTRAN` / `N_DOWNSAMPLING` / `DEPTH` / `HEADS` | `16` / `3` / `3` / `6` | generator shape — train and infer are passed the same values |

## CyTran is not a drop-in ResViT clone

`BENCHMARK.md` assumed the two repos share a codebase and that the ResViT adapter
would work with "CyTran's `options/base_options.py` data path". Reading the code,
that is wrong — they share only the pix2pix skeleton. Three things had to be
built, all of them handled by `setup`:

1. **The repo does not import**, in three separate places, all left by the same
   carve-out from a parent `pix2pix` package. `setup` sed-patches all three,
   idempotently:
   - `data/__init__.py` and `models/__init__.py` kept the absolute prefix
     (`importlib.import_module("pix2pix.data." + …)`), so `create_dataset()` and
     `create_model()` raise `ModuleNotFoundError` before anything runs → patched
     to `"data."` / `"models."`.
   - `models/cytran_model.py` and `models/cycle_gan_model.py` do
     `from util import ImagePool`, but `ImagePool` lives in `util/image_pool.py`
     and `util/__init__.py` is a bare docstring that re-exports nothing. That
     import only ever resolved as `pix2pix.util`'s `__init__` → patched to
     `from util.image_pool import ImagePool`. This one fires *after* option
     parsing, so it surfaces as a traceback the moment `train.py` looks up the
     model class.
2. **The stock loader reads Coltea DICOM.** `data/ct_dataset.py` unpickles a dict
   of DICOM paths from the Coltea-Lung-CT-100W release and rescales raw stored
   values with `x/1e3 - 1`. We supply [`cytran_vindr_dataset.py`](cytran_vindr_dataset.py),
   copied to `CyTran/data/vindr_dataset.py` and selected with
   `--dataset_mode vindr`; it reads the same pix2pix AB-PNG slices ResViT uses
   (`A` = NCCT, `B` = CECT) and hands the model `[-1,1]`.
3. **`test.py` cannot be used.** It is a hardcoded Coltea metrics script: it
   globs a DICOM tree, computes MAE/RMSE/SSIM in the **ARTERIAL→NATIVE**
   direction (the reverse of ours), and never writes an image.
   [`infer_cytran.py`](infer_cytran.py) loads the trained `G_A` weights into a
   bare `ConvTransformer` and does the forward pass itself.

The shims and the dataset file live at the **repo root** so that the delta
against upstream CyTran stays visible in one place; `setup` copies
`cytran_vindr_dataset.py` into `CyTran/data/` where the loader lookup expects it.

Two smaller notes: training is **single-stage** (no pretrain/fine-tune split like
ResViT, and no pretrained checkpoint to download), and CyTran's `test.py` /
`ct_dataset.py` still use the removed `np.float` alias — our path imports
neither, so they are left alone.

## What each stage does

| stage | action |
|---|---|
| `setup` | verifies the vendored `CyTran/` tree, patches the three broken imports left by the `pix2pix` carve-out, installs `data/vindr_dataset.py` |
| `prep` | our NIfTI → pix2pix AB-PNG slices via `prep_benchmark_data.py` (identical to ResViT's prep) |
| `train` | CyTran, `--direction AtoB` so `G_A` is NCCT→CECT |
| `infer` | `G_A` over the test slices → `<case>_<zzzz>_fake_B.npy` |
| `reassemble` | slices → per-case CECT NIfTI (HU, source grid) + `manifest.csv` |

**Direction.** `A` = NCCT, `B` = CECT, trained `AtoB`, so `G_A` is our generator
and `latest_net_G_A.pth` is the checkpoint that matters. `G_B` (CECT→NCCT) is
trained too — CyTran is cycle-consistent — but is not scored.

**Output range.** The generator's last layer is a plain conv, no `tanh`, so
outputs are not bounded. `infer_cytran.py` saves them raw as `.npy` and
`reassemble` clips with `--in_range neg1_1`, matching the `[-1,1]` the dataset
feeds in. (ResViT differs here: its `tensor2im` writes display PNGs, hence
`--in_range 0_255`.)

**Slice matching.** `reassemble` is given an explicit anchored pattern,
`^(?P<case>.+)_(?P<z>[0-9]{4})_fake_B\.npy$`, rather than the default heuristic.
Our `case_id`s are dotted DICOM UIDs
(`1.2.840.113619.2.278.3.717616.286.1587944968.878`) which the default handles
correctly, but a greedy prefix + fixed-width `z` + literal suffix cannot mis-split
under any case_id. If reassembly ever prints `no output slices matched` or
`only N/Z slices matched`, this is the first thing to check.

## Expect weaker pixel metrics than ResViT — by design

`models/cytran_model.py` is CycleGAN with `ConvTransformer` generators: GAN +
cycle + identity losses, and **no paired L1 term** even though our data is
paired. ResViT trains with `--lambda_A 100` on an aligned L1. So CyTran is
structurally disadvantaged on MSE/PCC/SSIM and may do relatively better on the
phase-fidelity columns, which is exactly the kind of contrast the benchmark is
meant to surface. Report it, don't "fix" it — changing their objective would stop
it being their model.

## Scoring

```bash
cd ../synthetic_CECT
python benchmark.py --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \
    --manifest ours=../out_synthesis_train/literature_baseline_l1_organ_curriculum/phase_infer/manifest.csv \
    --manifest resvit=../bench_ncct2cect/ResViT/results/vindr_nifti/manifest.csv \
    --manifest cytran=../bench_ncct2cect/CyTran/results/vindr_nifti/manifest.csv \
    --baseline ours --out analysis/bench_ncct2cect
# → analysis/bench_ncct2cect/master_table.md
```

`analysis/bench_ncct2cect` — NOT `analysis/benchmark`, which is this repo's *own*
loss/architecture ablation store (RUN_ALL.md §2). Also: you don't need to keep
re-passing `resvit`'s manifest each time — it's already cached in
`analysis/bench_ncct2cect/store` (see RUN_ALL.md, "results accumulate"). Add
`--manifest ours=...` once you've picked which of your own runs is `ours`;
until then, drop that line and score the external baselines on their own.

## Next module: SynDiff

Diffusion, so budget real GPU time. The pieces already exist:

- `prep_benchmark_data.py --format mat` writes `data_<phase>_NCCT.mat` /
  `data_<phase>_CECT.mat`, HDF5 with variable `data_fs`, shape `(N,256,256)` in
  `[0,1]` — the layout SynDiff's `LoadDataSet` expects. Needs `h5py`.
- Its own env (`torch>=1.7`, needs `python3.x-dev` per its README) — do **not**
  merge with CFPS-Diff's `torch==2.0.0+cu118`.
- Reassemble with `--in_range neg1_1` and an anchored `--pattern` for whatever
  suffix its sampler writes.

The open question to answer first, by reading its code the way we read CyTran's:
**does its inference script write per-slice files with recoverable indices, or
only a batched array?** If batched, the adapter must carry the `slice_index.csv`
row order through — that ordering is already deterministic in the `.mat` writer
(rows are appended in `phase → case → z` order), so it can be reconstructed, but
it must be done deliberately rather than assumed.

Then CFPS-Diff last: roughest repo, reads NIfTI natively, multiphase output —
export only the venous volume.

# Running SynDiff on our VinDr NCCT→CECT data

Third benchmark module, after ResViT and CyTran. Same contract: train on the
shared split, emit per-case synthetic CECT NIfTI (HU, source grid) +
`manifest.csv` for `../synthetic_CECT/benchmark.py`.

```bash
SPLIT=../synthetic_CECT/splits/split.json GPU=0 ./run_syndiff.sh all
```

## What you need

1. **A GPU.** There is no CPU path — `train.py` initialises `torch.distributed`
   with the NCCL backend even for a single process, and the backbones
   JIT-compile CUDA kernels at import.
2. **A working nvcc + ninja + `pythonX.Y-dev`.** See below; `setup` preflights
   all three so you fail in five seconds instead of mid-epoch.
3. **The shared split** `../synthetic_CECT/splits/split.json` (seed 42), plus
   `pip install -r requirements_syndiff.txt` in a dedicated env.

## Stages

| stage | what it does |
|---|---|
| `setup` | verifies the vendored `SynDiff/` tree and preflights the JIT-CUDA toolchain |
| `prep` | our NIfTI → `data_{train,val,test}_{NCCT,CECT}.mat` (HDF5, key `data_fs`) |
| `train` | adversarial diffusion, `contrast1=NCCT` → `contrast2=CECT` |
| `infer` | `gen_diffusive_2` over the test slices → per-slice `*_fake_B.npy` |
| `reassemble` | slices → per-case CECT NIfTI (HU, source grid) + `manifest.csv` |

Run them individually or as `all`. Every variable is overridable from the
environment (`NUM_EPOCH`, `BATCH`, `GPU`, `NAME`, `PORT`, …).

## The three things that make SynDiff different

### 1. It JIT-compiles CUDA at import

`backbones/up_or_down_sampling.py` does `from utils.op import upfirdn2d`, and
`utils/op/upfirdn2d.py` calls `torch.utils.cpp_extension.load(...)` at module
scope — it shells out to **ninja** and **nvcc** and needs **`Python.h`**. Miss
any one and the import explodes. `setup` checks for all three:

```
[run_syndiff:setup] preflight the JIT-CUDA toolchain
  MISSING: ninja        (pip install ninja)
  ok: Python.h in /usr/include/python3.10
```

Verify by hand with `cd SynDiff && python -c 'from utils.op import upfirdn2d'`.
The first successful import takes a minute or two while it builds; after that
it is cached in `~/.cache/torch_extensions`.

### 2. `SIZE` must be 256

`dataset.py::LoadDataSet` pads every slice to a hardcoded 256×256 with
`pad = (256 - shape) / 2`. At `SIZE > 256` that pad is negative and `np.pad`
raises; at `SIZE < 256` it silently zero-borders every slice, shifting the
anatomy so the reassembled volume no longer aligns with the source grid.
`run_syndiff.sh` refuses anything else.

### 3. Its `test.py` is unusable — we ship `infer_syndiff.py`

Three independent blockers, all in `sample_and_test`:

- `crop = transforms.CenterCrop((256, 152))` — the IXI/BRATS field of view,
  hardcoded. The cropped `(256,152)` result is then assigned into a preallocated
  `(256,256,N)` buffer, which is a plain `ValueError` on **any** dataset. It
  would also throw away a third of the CT.
- `fake_sample = fake_sample / fake_sample.max()` — per-slice renormalisation.
  That destroys the global intensity scale, so slices can no longer share one HU
  window. Fatal for volume reconstruction.
- outputs are `.jpg` triptychs plus one `im_syn.mat`, none of which carries the
  `case_id`/`z` needed to rebuild a volume.

[`infer_syndiff.py`](infer_syndiff.py) imports upstream's own
`sample_from_model` / `Posterior_Coefficients` / `load_checkpoint` so the
diffusion math cannot drift, and only replaces the I/O around them.

## Details worth knowing

- **Direction.** Our prep writes `contrast1 = NCCT` (source) and
  `contrast2 = CECT` (target). SynDiff trains both directions at once;
  `gen_diffusive_2` is the NCCT→CECT one, so that is the checkpoint inference
  loads. `gen_diffusive_1` (CECT→NCCT) is ignored.
- **The transpose.** `LoadDataSet` applies `np.transpose(data, (0,2,1))` to any
  3-D `.mat` — a MATLAB v7.3 column-major convention. Our `.mat` is written by
  h5py and does not need it, but the loader applies it unconditionally, so the
  model sees every slice transposed. Both contrasts get the same treatment, so
  training is self-consistent; `infer_syndiff.py` transposes back on the way out.
  Verified round-trip: `(loaded * 0.5 + 0.5).T == prep input`, exactly.
- **Checkpoints** land in `SynDiff/results/<NAME>/gen_diffusive_2_<epoch>.pth`
  and are written only every `SAVE_CKPT_EVERY` (default 10) epochs. `infer`
  picks the newest automatically; pin one with `CKPT_EPOCH=30`.
- **Output range.** Slices are saved as raw `[-1,1]` `.npy` with no per-slice
  rescaling, so reassembly uses `--in_range neg1_1`.
- **DDP port.** `train.py` needs a free `MASTER_PORT`; override with `PORT=6037`
  if you run two experiments on one box.
- **Resuming.** `--save_content` is on, so `results/<NAME>/content.pth` lets an
  interrupted run pick up where it stopped — unlike the ResViT test stage, which
  restarts from slice 0.

## Scoring

```bash
cd ../synthetic_CECT && python benchmark.py \
    --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \
    --manifest syndiff=../ncct2cect/SynDiff/results/vindr_nifti/manifest.csv \
    --baseline ours --out analysis/benchmark
```

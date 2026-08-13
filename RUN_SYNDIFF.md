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

**`nvcc` is usually installed but not on `PATH`.** A box that runs torch fine
often ships only the CUDA *runtime*, with the compiler sitting in
`/usr/local/cuda*/bin`. `run_syndiff.sh` searches `PATH`, `$CUDA_HOME/bin` and
`/usr/local/cuda*/bin`, and exports `CUDA_HOME`/`PATH` itself when it finds one.
If it genuinely is not installed, get a toolkit matching the CUDA your torch was
built for (the preflight prints both):

```bash
python -m pip install ninja                 # ninja goes in the active venv
conda install -c nvidia cuda-nvcc           # no root
sudo apt install nvidia-cuda-toolkit        # needs root; check the version it gives
```

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
- **Resuming.** On by default (`RESUME=1`), writing `results/<NAME>/content.pth`
  after every epoch (`SAVE_CONTENT_EVERY=1`). Set `RESUME=0` to force a fresh
  start. Upstream's `train.py` ignores `content.pth` unless `--resume` is passed
  and starts from epoch 0 otherwise, so before this was wired up a crash-relaunch
  loop would sit in "epoch 0" indefinitely while looking like a running job.

## Throughput — what this costs and why

SynDiff is by a wide margin the most expensive model in this benchmark, and the
reason is architectural rather than "diffusion is slow". With `NUM_TIMESTEPS=4`
training samples ONE random timestep per iteration; the chain is never unrolled
except at inference. The cost is that **eight networks** are trained at once —
two NCSN++ diffusive generators, two ResNet CycleGAN translators, two
time-conditioned discriminators and two cycle discriminators — in both
directions simultaneously. One iteration consumes a single 256×256 slice through
roughly 4 NCSN++ forwards, 8 translator forwards, 12 discriminator forwards and
3 backward passes, plus an R1 double-backward every `--lazy_reg` steps. ResViT or
CyTran spends one generator forward and one discriminator forward on the same
slice.

The training loop as vendored also carried several avoidable costs, all fixed in
this tree (see the `SPEED:` comments in `SynDiff/train.py`):

| what | why it cost | fix |
|---|---|---|
| `torch.autograd.set_detect_anomaly(True)` in the hot loop | global, sticky, never disabled → per-node stack traces + NaN checks on *every* backward for the whole run, ~2-4x | removed; opt in with `--detect_anomaly` |
| generator outputs not detached in the D steps | `errD_fake.backward()` ran through both UNets and both translators, and the gradients were then discarded by `zero_grad()` | both D blocks now run their generator forwards under `no_grad()` |
| translators run a third time for the cycle-D | identical inputs, identical weights, identical outputs | reuse the tensors from the diffusive-D block |
| redundant host→device copies of `x1`/`x2` | already on the device | removed |
| `num_workers=4` over an in-RAM `TensorDataset` | one IPC pickle per sample with no I/O to overlap | `NUM_WORKERS=0` |
| validation every epoch | full diffusion sampler over every val slice, both directions, `batch=1` | `VAL_EVERY=10`; the final epoch is always validated |
| `cudnn.benchmark` off | every shape is fixed at 256×256 | enabled, plus TF32 on Ampere+ |

The training log now prints `s/it`, an epoch ETA and a run ETA every 100
iterations, so "expensive architecture" and "broken run" are distinguishable
without timing the output by hand.

Also fixed: `val_l1_loss`/`val_psnr_values` were allocated with `num_epoch` rows
but indexed by an epoch loop running to `num_epoch` inclusive, so the final epoch
of any run raised `IndexError` *after* the entire training cost had been paid.

## Scoring

```bash
cd ../synthetic_CECT && python benchmark.py \
    --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \
    --manifest syndiff=../ncct2cect/SynDiff/results/vindr_nifti/manifest.csv \
    --baseline ours --out analysis/benchmark
```

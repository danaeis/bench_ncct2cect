# Running CycleGAN on our VinDr data (remote GPU server)

Fourth benchmark module, after ResViT, CyTran and SynDiff. Same contract: train
on the shared split, emit per-case synthetic CECT NIfTI (HU, source grid) +
`manifest.csv` for `../synthetic_CECT/benchmark.py`.

```bash
SPLIT=../synthetic_CECT/splits/split.json GPU=0 ./run_cyclegan.sh all
```

## Which CycleGAN?

`CycleGAN/` is the official PyTorch
[junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix),
vendored as a plain directory like the other baselines.

⚠️ The pre-existing `UnpairedImageTranslation/` in this repo is **not** used. It
is Shizuo Kaji's **Chainer** reimplementation of CycleGAN — Chainer has been
unmaintained since 2019, and getting it running would mean resurrecting
`chainer` + a CUDA-10.2 `cupy` wheel (both are sitting unused in `vindr_ds/`).
The PyTorch original gives the same model on a maintained stack, and shares its
codebase lineage with ResViT and CyTran, so one adapter pattern covers all three.

## Why it is in the benchmark

CycleGAN is the **unpaired** baseline. Our slices are perfectly paired on disk,
but `--dataset_mode unaligned` draws A and B independently, so the model never
sees a matched pair — it is trained purely by cycle-consistency plus two
discriminators. That is the point: it measures what the unpaired objective costs
on data where pairing *is* available, which is the honest control for every
"we don't need registered pairs" claim in the literature.

Do not "fix" this by switching to aligned mode. That is just pix2pix, and ResViT
already covers the paired-adversarial corner.

## Stages

| stage | what it does |
|---|---|
| `setup` | verifies the vendored tree, applies the wandb compat shim |
| `prep` | our NIfTI → `{train,val,test}A/` (NCCT) + `{train,val,test}B/` (CECT) |
| `train` | `--model cycle_gan`, A=NCCT → B=CECT, plus the B→A cycle it needs |
| `test` | `netG_A` over the test slices → `<case>_<zzzz>_fake.png` |
| `reassemble` | slices → per-case CECT NIfTI (HU, source grid) + `manifest.csv` |

Tunables (env): `SPLIT`, `DATAROOT`, `SIZE` (256), `GPU`, `NAME`, `NITER` /
`NITER_DECAY`, `BATCH`, `WORKERS`, `LAMBDA_IDT`, `SAVE_EPOCH_FREQ`, `RESUME`,
`EPOCH_COUNT`, `OUT_NIFTI`, `MIN_TISSUE_FRAC`.

## Resuming after a crash or a reboot

`RESUME=1` is the **default**: re-running `./run_cyclegan.sh train` continues
from `checkpoints/<name>/latest_net_G_A.pth` rather than restarting at epoch 1
and overwriting the checkpoints.

The restart epoch comes from the newest **numbered** checkpoint
(`<epoch>_net_G_A.pth`, every `SAVE_EPOCH_FREQ`), while `latest_net_*.pth` is
refreshed every `--save_latest_freq` iterations, so the reloaded weights can be
slightly ahead of the epoch the lr schedule resumes at. Safe, but not
bit-identical to an uninterrupted run — pin `EPOCH_COUNT=N` to be exact, or
`RESUME=0` to start clean.

Smoke-test before committing to a full run:

```bash
./run_cyclegan.sh setup && ./run_cyclegan.sh prep
NITER=1 NITER_DECAY=0 SAVE_EPOCH_FREQ=1 ./run_cyclegan.sh train
./run_cyclegan.sh test && ./run_cyclegan.sh reassemble
```

## Details that took a pass through the source to get right

- **A new prep format.** CycleGAN wants two directories per phase (`trainA`,
  `trainB`), not ResViT's single AB-concatenated PNG, so
  `prep_benchmark_data.py` gained `--format cyclegan`. Same windowing, same
  resize, same `slice_index.csv` — only the on-disk layout differs. `out_ref`
  points at the **A**-side file, because CycleGAN names its output after the
  input it was fed.
- **`--preprocess none --no_flip`.** The default `resize_and_crop` would rescale
  256→286 and random-crop back to 256. Random crops break the 1:1
  correspondence with the source grid that reassembly depends on, and horizontal
  flips mirror left/right abdominal anatomy (liver ↔ spleen), which is not a
  valid CT augmentation.
- **No `--gpu_ids`.** This HEAD dropped it: `util.init_ddp()` takes `cuda:0` when
  CUDA is visible and CPU otherwise. The runner therefore selects the GPU with
  `CUDA_VISIBLE_DEVICES`, exactly like `run_syndiff.sh`. `GPU=-1` masks every
  device, i.e. CPU.
- **`--num_test` defaults to 50** and would silently truncate the test set to
  the first 50 slices — the same trap as ResViT's `--how_many`. The runner passes
  1000000.
- **Inference uses `--model test --model_suffix _A`**, which loads only
  `latest_net_G_A.pth` and reads `testA/` through `dataset_mode single`. Running
  `--model cycle_gan` at test time would also demand `testB` and write six
  visuals per slice. Note the resulting suffix is **`_fake`**, not `_fake_B` —
  `TestModel`'s visual is named `fake` — so reassembly matches
  `^(?P<case>.+)_(?P<z>[0-9]{4})_fake\.png$`.
- **Compat shim (idempotent, committed).** `util/visualizer.py` does a top-level
  `import wandb`, though wandb is only touched when `--use_wandb` is passed.
  The shim makes that import conditional on the flag, so a large optional
  dependency stays optional.
- **`WORKERS` on macOS.** `--preprocess none` installs a `transforms.Lambda`,
  which the *spawn* start method cannot pickle. Linux forks its dataloader
  workers, so the default `WORKERS=4` is fine on the server; set `WORKERS=0` if
  you ever smoke-test on a Mac.
- **Output range.** `tensor2im` writes display PNGs in `[0,255]` →
  `--in_range 0_255`, same as ResViT.
- **Identity loss** stays at upstream's `0.5`. It discourages the generator from
  shifting HU where no enhancement is expected, which is a genuine regulariser
  here and not just a photographic colour trick. `LAMBDA_IDT=0` to ablate it.

## Scoring

```bash
cd ../synthetic_CECT
python benchmark.py --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \
    --manifest cyclegan=../ncct2cect/CycleGAN/results/vindr_nifti/manifest.csv \
    --baseline ours --out analysis/benchmark
# → analysis/benchmark/master_table.md
```

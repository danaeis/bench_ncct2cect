# Running the NCCT→CECT benchmark — ordered steps

Two repos: **`ncct2cect`** (competing models) and **`synthetic_CECT`** (scoring +
our model). They meet at the **manifest contract**: any model emits synthetic CECT
NIfTI on each test case's source grid (HU) + a CSV
`gen_path,real_path,mask_path,target_phase`; `benchmark.py` scores them all.

Do the stages in order. **Stage 3 gives a real table with zero training** (scores
our 5 existing runs) — do it before touching any external repo.

---

## Stage 0 — Get code onto the GPU server

```bash
# bench_ncct2cect: the four baseline repos are vendored, a plain clone is enough
git clone https://github.com/danaeis/bench_ncct2cect
#   or, if already cloned:
git -C bench_ncct2cect pull

# synthetic_CECT scoring code: see the "Getting the scoring code there" note at
# the bottom — metrics.py / benchmark.py / dump_split.py must be present.
```

## Stage 1 — Environments (isolate per repo; deps conflict)

```bash
# Scoring env (light): numpy scipy nibabel pandas xgboost  [scikit-image optional]
conda create -n bench python=3.10 -y && conda activate bench
pip install numpy scipy nibabel pandas xgboost scikit-image

# Per-model envs — do NOT merge (SynDiff torch>=1.7 vs CFPS-Diff torch==2.0+cu118):
#   ResViT / CyTran : follow each README (pix2pix-style, torch>=1.7)
#   SynDiff         : README env (torch>=1.7, needs python3.x-dev)
#   CFPS-Diff       : conda env python=3.8, torch==2.0.0+cu118  (see its README)
```

## Stage 2 — The shared split (once)

```bash
cd synthetic_CECT
python dump_split.py --seg_suffix _seg_full          # → splits/split.json
```
Every external adapter reads this file. Same 20 test cases for everyone.

## Stage 3 — Validate the harness AND get the first table (no training)

```bash
cd synthetic_CECT
python test_metrics.py                                # unit gate, must pass

python benchmark.py \
  --runs_dir ../out_synthesis_train \
  --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \
  --baseline l1_organ_curriculum \
  --out analysis/benchmark
# → analysis/benchmark/master_table.md  (our 5 models, full metric suite)
```
**Gate:** the phase columns (phase-acc, gen-prob, featHU) must reproduce
`analyze_runs.py` (organ_curriculum featHU ≈ 15.74); the new pixel columns
(MSE/PCC + windowed SSIM) are the additions. If phase numbers drift, stop and
diff against `analyze_runs.py` before trusting anything downstream.

## Stage 4 — External models, one at a time

Order = cheapest/closest first, so the adapter pattern is proven before the hard
repos. **Each model needs a small adapter you write once** — the repos train on
their own data formats, not ours. The adapter does 3 things:
(a) read `splits/split.json`; (b) convert our NIfTI → the repo's input, train,
infer; (c) export synthetic CECT **NIfTI on the source grid (HU)** + a manifest.
Reuse `infer_volume.py`'s stitching when a repo outputs slices/patches.

**Order and per-repo notes (from BENCHMARK.md):**

1. **ResViT** — pix2pix-style aligned pairs; `train.py` / `test.py`. Closest to a
   paired setup, cheapest. Write `ResViT/adapter_resvit.py` first; it becomes the
   template.
2. **CyTran** — its README claims the ResViT/pix2pix codebase family, but only
   the skeleton is shared: the repo does not even import standalone, its loader
   reads Coltea DICOM, and its `test.py` writes no images. Reuses ResViT's
   *prepped slices*, not its adapter. Driven by `run_cytran.sh` — see
   [RUN_CYTRAN.md](RUN_CYTRAN.md).
3. **SynDiff** — diffusion; wants **`.mat`** `(#img,W,H)` in [0,1]. Adapter must
   convert slices to `.mat` and generated slices back to HU NIfTI.
4. **CFPS-Diff** — roughest repo; `main/train_CFPS-Diff_gen.py`,
   `main/Inference.py`; multiphase output → export only the venous phase. Do last.

Per model, after its adapter runs:
```bash
# (inside that model's env) train + infer → produces <out>/manifest.csv
# then score it (in the bench env), appended to the growing table:
cd synthetic_CECT
python benchmark.py --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \
  --manifest ours=../out_synthesis_train/literature_baseline_l1_organ_curriculum/phase_infer/manifest.csv \
  --manifest resvit=../ncct2cect/ResViT/<out>/manifest.csv \
  --baseline ours --out analysis/benchmark
```

## Stage 5 — Final master table

Run Stage 4's `benchmark.py` once with **all** manifests (ours + 4 external).
Output `analysis/benchmark/master_table.md`: every model × full suite, paired
per-case tests vs ours. Lead with organ-region + phase fidelity; report pixel
metrics with the blur caveat (PROJECT_PLAN.md).

---

## Reality checks

- **The adapters are the real remaining work.** Harness + vendored repos are done;
  each external repo needs its ~100-line data adapter before it can train on our
  split. ResViT/CyTran share one; SynDiff/CFPS-Diff need format conversion.
- **Same-data caveat:** retraining these on our data at our scale will not match
  their papers' numbers. That is expected — it's a controlled comparison, not a
  reproduction. State it in the writeup.
- **CFPS-Diff/SynDiff are diffusion** — slow to train; budget GPU time. This is
  why they are last.

## Getting the scoring code to the server

`synthetic_CECT` has remote `git@github.com:danaeis/synthetic_CECT.git` but a lot
of uncommitted work from earlier tasks. Options, your call:
- **Targeted:** commit only the benchmark files
  (`metrics.py benchmark.py dump_split.py test_metrics.py RUN_BENCHMARK.md`) on a
  branch, push, cherry-pull on the server.
- **Everything:** commit all pending changes (organ-weighting, curriculum, sample
  grids, analysis scripts) and push — bigger, may diverge from server state.
- **scp/rsync** the files directly if the server copy is ahead of the remote.

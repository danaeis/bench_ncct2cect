#!/usr/bin/env bash
# =============================================================================
# run_cyclegan.sh — end-to-end CycleGAN NCCT->CECT run on OUR VinDr data.
#
# Fourth benchmark module. Same contract as the others: train on the shared
# split, emit per-case synthetic CECT NIfTI (HU, source grid) + manifest.csv
# for ../synthetic_CECT/benchmark.py.
#
#   SPLIT=../synthetic_CECT/splits/split.json GPU=0 ./run_cyclegan.sh all
#
# Stages (run one, or `all`):
#   setup       verify the vendored CycleGAN tree
#   prep        our NIfTI -> <phase>A/ (NCCT) + <phase>B/ (CECT) slice dirs
#   train       CycleGAN, A=NCCT -> B=CECT (plus the B->A cycle it needs)
#   test        netG_A over the test slices -> <case>_<zzzz>_fake.png
#   reassemble  slices -> per-case CECT NIfTI (HU, source grid) + manifest.csv
#   all         every stage above, in order
#
# What makes CycleGAN different from ResViT/CyTran, which share its codebase:
#
#   * It is the **unpaired** baseline. Our slices are perfectly paired on disk,
#     but `--dataset_mode unaligned` samples B independently of A, so the model
#     never sees a matched pair and is trained purely by cycle-consistency +
#     two discriminators. That is the point of including it: it measures what
#     the unpaired objective costs on data where pairing is actually available.
#     Do NOT "fix" this by switching to aligned mode — that would just be
#     pix2pix, which ResViT already covers.
#   * Its dataset layout is two directories per phase (trainA/trainB), not one
#     AB-concatenated PNG -> `prep_benchmark_data.py --format cyclegan`.
#   * `--preprocess none --no_flip`: our slices are already exactly SIZE, and
#     the default `resize_and_crop` would rescale 256->286 and random-crop back.
#     Random crops break the 1:1 correspondence with the source grid that
#     reassembly relies on, and horizontal flips mirror left/right abdominal
#     anatomy (liver <-> spleen), which is not a valid augmentation for CT.
#   * Inference uses `--model test --model_suffix _A`, which loads only
#     `latest_net_G_A.pth` and reads `testA/` through `dataset_mode single`.
#     Running `--model cycle_gan` at test time would also demand testB and emit
#     six visuals per slice; we need exactly one, named after its A input.
#     Note the suffix is `_fake`, not `_fake_B` (TestModel's visual is `fake`).
# =============================================================================
set -euo pipefail

# ---- config (override via env) ----------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CYCLEGAN="$HERE/CycleGAN"

SPLIT="${SPLIT:-}"                                              # shared case split
DATAROOT="${DATAROOT:-$CYCLEGAN/datasets/vindr}"                # prepped slices
SIZE="${SIZE:-256}"
# Drop train/val slices with < this fraction of >0.1 voxels (0 = keep every
# slice, the default). Test is never filtered, so reconstruction stays complete.
MIN_TISSUE_FRAC="${MIN_TISSUE_FRAC:-0.0}"
GPU="${GPU:-0}"                                                 # gpu id; -1 = cpu
NAME="${NAME:-vindr_cyclegan}"
NITER="${NITER:-25}"             # --n_epochs        (epochs at full lr)
NITER_DECAY="${NITER_DECAY:-25}" # --n_epochs_decay  (epochs of linear decay)
BATCH="${BATCH:-1}"
# DataLoader workers (--num_threads). `--preprocess none` installs a
# transforms.Lambda, which the *spawn* start method cannot pickle; that only
# bites on macOS/Windows, since Linux forks its workers. Set WORKERS=0 there.
WORKERS="${WORKERS:-4}"
SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-5}"
# Identity loss. Upstream's default is 0.5 and it exists to preserve colour
# composition in photo translation. It is a genuine regulariser for CT too --
# it discourages the generator from shifting HU where no contrast is expected --
# so we keep the default rather than silently zeroing it.
LAMBDA_IDT="${LAMBDA_IDT:-0.5}"
OUT_NIFTI="${OUT_NIFTI:-$CYCLEGAN/results/vindr_nifti}"

PY="${PYTHON:-python3}"

# This CycleGAN HEAD has no --gpu_ids: util.init_ddp() picks cuda:0 when CUDA is
# visible and CPU otherwise, so the GPU is selected by masking the device list.
# GPU=-1 therefore means "hide every GPU", i.e. run on CPU.
if [ "$GPU" = "-1" ]; then CUDA_MASK=""; else CUDA_MASK="$GPU"; fi

# Never route loopback through the site proxy: it answers CONNECT with 403 and
# the client stalls on retries.
export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

log() { printf '\n\033[1;36m[run_cyclegan:%s]\033[0m %s\n' "$1" "$2"; }

# `sed -i` takes a mandatory suffix on BSD/macOS and none on GNU; write through a
# temp file so this works on either.
sed_inplace() {
  local expr="$1" file="$2"
  if ! sed "$expr" "$file" > "$file.tmp"; then
    rm -f "$file.tmp"
    echo "ERROR: sed failed on $file: $expr" >&2
    exit 1
  fi
  mv "$file.tmp" "$file"
}

# CycleGAN/ is vendored, so this patch is committed and this is normally a no-op.
# Kept, and called from every stage that runs CycleGAN code, so a fresh checkout
# of upstream dropped in here still works. Idempotent.
apply_shims() {
  # util/visualizer.py does a top-level `import wandb`, but wandb is only ever
  # touched when --use_wandb is passed (we never pass it). That turns a large,
  # network-registering, optional dependency into a hard one for every run.
  # Make the import conditional on the flag actually being present.
  if grep -q '^import wandb$' "$CYCLEGAN/util/visualizer.py"; then
    sed_inplace 's|^import wandb$|import importlib, sys as _sys; wandb = importlib.import_module("wandb") if "--use_wandb" in _sys.argv else None  # shim: optional dep|' \
      "$CYCLEGAN/util/visualizer.py"
    echo "  shim: visualizer.py lazy wandb import"
  fi
}

# Locate the shared case split. Honour $SPLIT if set, else try the known layouts
# of the synthetic_CECT sibling repo. Sets $SPLIT or exits with the paths tried.
resolve_split() {
  if [ -n "$SPLIT" ]; then
    [ -f "$SPLIT" ] || { echo "ERROR: split not found: $SPLIT" >&2; exit 1; }
    return
  fi
  local candidates=(
    "$HERE/../synthetic_CECT/splits/split.json"
    "$HERE/../synthetic_CECT/benchmark/split.json"
    "$HERE/../diff_synthetic_CECT/splits/split.json"
  )
  local c
  for c in "${candidates[@]}"; do
    if [ -f "$c" ]; then SPLIT="$c"; echo "  split: $SPLIT"; return; fi
  done
  { echo "ERROR: no split.json found. Tried:"
    printf '  %s\n' "${candidates[@]}"
    echo "Set it explicitly:  SPLIT=/path/to/split.json $0 prep"; } >&2
  exit 1
}

# ---- stages -----------------------------------------------------------------
do_setup() {
  log setup "check CycleGAN tree"
  [ -f "$CYCLEGAN/train.py" ] || { echo "ERROR: CycleGAN/ is missing or empty at $CYCLEGAN" >&2; exit 1; }
  echo "  present: $CYCLEGAN"

  log setup "apply compat shims (idempotent)"
  apply_shims
  echo
  echo "  deps: pip install -r $HERE/requirements_cyclegan.txt  (in a dedicated env)"
  echo "  NB: this is the PyTorch junyanz/pytorch-CycleGAN-and-pix2pix repo."
  echo "      The pre-existing UnpairedImageTranslation/ is a Chainer CycleGAN"
  echo "      and is NOT used by this runner."
}

do_prep() {
  log prep "our NIfTI -> {train,val,test}{A,B}/ slice dirs at $DATAROOT"
  resolve_split
  "$PY" "$HERE/prep_benchmark_data.py" --split "$SPLIT" \
      --format cyclegan --out "$DATAROOT" --size "$SIZE" \
      --min_tissue_frac "$MIN_TISSUE_FRAC"
}

do_train() {
  [ -d "$DATAROOT/trainA" ] || { echo "ERROR: $DATAROOT/trainA missing (run prep first)" >&2; exit 1; }
  log train "CycleGAN A=NCCT -> B=CECT ($NITER + $NITER_DECAY epochs)"
  apply_shims
  cd "$CYCLEGAN"
  CUDA_VISIBLE_DEVICES="$CUDA_MASK" "$PY" train.py --dataroot "$DATAROOT" --name "$NAME" \
      --model cycle_gan --dataset_mode unaligned --direction AtoB \
      --input_nc 1 --output_nc 1 --load_size "$SIZE" --crop_size "$SIZE" \
      --preprocess none --no_flip \
      --n_epochs "$NITER" --n_epochs_decay "$NITER_DECAY" \
      --lambda_identity "$LAMBDA_IDT" \
      --batch_size "$BATCH" --num_threads "$WORKERS" \
      --save_epoch_freq "$SAVE_EPOCH_FREQ" \
      --checkpoints_dir checkpoints/ --no_html
}

do_test() {
  local w="$CYCLEGAN/checkpoints/$NAME/latest_net_G_A.pth"
  [ -f "$w" ] || { echo "ERROR: generator weights missing: $w (run train first)" >&2; exit 1; }
  log test "netG_A over the test slices -> *_fake.png"
  apply_shims
  cd "$CYCLEGAN"
  # dataset_mode single reads a flat dir of images, so dataroot points AT testA.
  # --num_test defaults to 50 and would silently truncate the test set.
  CUDA_VISIBLE_DEVICES="$CUDA_MASK" "$PY" test.py --dataroot "$DATAROOT/testA" --name "$NAME" \
      --model test --model_suffix _A --dataset_mode single \
      --input_nc 1 --output_nc 1 --load_size "$SIZE" --crop_size "$SIZE" \
      --preprocess none --no_flip --no_dropout \
      --num_test 1000000 --eval --epoch latest --num_threads "$WORKERS" \
      --results_dir results/ --checkpoints_dir checkpoints/
}

do_reassemble() {
  log reassemble "slices -> per-case CECT NIfTI (HU) + manifest.csv"
  # tensor2im writes display PNGs in [0,255] -> in_range 0_255. TestModel names
  # its single visual `fake`, so the suffix is _fake (not _fake_B like ResViT).
  "$PY" "$HERE/reassemble_nifti.py" \
      --index "$DATAROOT/slice_index.csv" \
      --slices_dir "$CYCLEGAN/results/$NAME/test_latest/images" \
      --pattern '^(?P<case>.+)_(?P<z>[0-9]{4})_fake\.png$' \
      --in_range 0_255 --out "$OUT_NIFTI"
  echo
  echo "  manifest -> $OUT_NIFTI/manifest.csv"
  echo "  score it: cd ../synthetic_CECT && python benchmark.py \\"
  echo "      --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \\"
  echo "      --manifest cyclegan=$OUT_NIFTI/manifest.csv --baseline ours --out analysis/benchmark"
}

# ---- dispatch ---------------------------------------------------------------
stage="${1:-all}"
case "$stage" in
  setup)      do_setup ;;
  prep)       do_prep ;;
  train)      do_train ;;
  test)       do_test ;;
  reassemble) do_reassemble ;;
  all)        do_setup; do_prep; do_train; do_test; do_reassemble ;;
  *) echo "usage: $0 [setup|prep|train|test|reassemble|all]" >&2; exit 2 ;;
esac
log "$stage" "done"

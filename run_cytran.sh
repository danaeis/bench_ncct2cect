#!/usr/bin/env bash
# =============================================================================
# run_cytran.sh — end-to-end CyTran NCCT->CECT run on OUR VinDr data.
#
# Second benchmark module, after ResViT. Same contract: train on the shared
# split, emit per-case synthetic CECT NIfTI (HU, source grid) + manifest.csv for
# ../synthetic_CECT/benchmark.py.
#
#   SPLIT=../synthetic_CECT/splits/split.json GPU=0 ./run_cytran.sh all
#
# Stages (run one, or `all`):
#   setup       verify the vendored CyTran tree, fix its two broken package imports,
#               install our VinDr dataset module into data/
#   prep        our NIfTI -> pix2pix AB-PNG slices (train/val/test) on the split
#   train       CyTran (ConvTransformer CycleGAN), A=NCCT -> B=CECT
#   infer       G_A over the test slices -> per-slice *_fake_B.npy
#   reassemble  slices -> per-case CECT NIfTI (HU, source grid) + manifest.csv
#   all         every stage above, in order
#
# CyTran is NOT a drop-in ResViT clone (they only share the pix2pix skeleton):
#   * its stock loader unpickles Coltea DICOM paths        -> we add data/vindr_dataset.py
#   * data/ and models/ import a `pix2pix.` package that   -> sed shim at setup
#     does not exist in the standalone repo (import crash)
#   * its test.py is a hardcoded Coltea metrics script that -> we use infer_cytran.py
#     writes no images and runs the reverse direction
# Training is single-stage: unlike ResViT there is no pretrain/fine-tune split.
# =============================================================================
set -euo pipefail

# ---- config (override via env) ----------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CYTRAN="$HERE/CyTran"

SPLIT="${SPLIT:-}"                                              # shared case split
DATAROOT="${DATAROOT:-$CYTRAN/datasets/vindr}"                  # prepped slices
SIZE="${SIZE:-256}"             # must be divisible by 2^N_DOWNSAMPLING
GPU="${GPU:-0}"                 # gpu id; -1 = cpu
NAME="${NAME:-vindr_cytran}"
NITER="${NITER:-25}"            # epochs at full lr      (--n_epochs)
NITER_DECAY="${NITER_DECAY:-25}" # epochs of lr decay    (--n_epochs_decay)
BATCH="${BATCH:-2}"
LR="${LR:-0.0001}"              # CyTran's default; lower than ResViT's
OUT_NIFTI="${OUT_NIFTI:-$CYTRAN/results/vindr_nifti}"
# train.py only writes latest_net_*.pth on epochs divisible by this, so the TOTAL
# epoch count must be a multiple of it or the run ends with no `latest`
# checkpoint. Set SAVE_EPOCH_FREQ=1 for short smoke tests.
SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-5}"
CKPT_EPOCH="${CKPT_EPOCH:-latest}"   # which checkpoint infer loads

# generator shape — training and inference MUST agree on these
NGF_CYTRAN="${NGF_CYTRAN:-16}"
N_DOWNSAMPLING="${N_DOWNSAMPLING:-3}"
DEPTH="${DEPTH:-3}"
HEADS="${HEADS:-6}"

PY="${PYTHON:-python3}"
SLICES="$CYTRAN/results/$NAME/images"
if [ "$GPU" = "-1" ]; then DEVICE=cpu; else DEVICE=cuda; fi

log() { printf '\n\033[1;36m[run_cytran:%s]\033[0m %s\n' "$1" "$2"; }

# `sed -i` takes a mandatory suffix on BSD/macOS and none on GNU; write through a
# temp file so setup works on either. Fails loudly rather than leaving the source
# half-patched and a .tmp behind.
sed_inplace() {
  local expr="$1" file="$2"
  if ! sed "$expr" "$file" > "$file.tmp"; then
    rm -f "$file.tmp"
    echo "ERROR: sed failed on $file: $expr" >&2
    exit 1
  fi
  mv "$file.tmp" "$file"
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
  log setup "check CyTran tree"
  [ -f "$CYTRAN/train.py" ] || { echo "ERROR: CyTran/ is missing or empty at $CYTRAN" >&2; exit 1; }
  echo "  present: $CYTRAN"

  log setup "apply compat shims (idempotent)"
  # The standalone repo was carved out of a parent `pix2pix` package but kept the
  # absolute import prefix, so create_dataset()/create_model() raise
  # ModuleNotFoundError before anything runs.
  if grep -q '"pix2pix\.data\."' "$CYTRAN/data/__init__.py"; then
    sed_inplace 's#"pix2pix\.data\." + dataset_name#"data." + dataset_name#' "$CYTRAN/data/__init__.py"
    echo "  patched data/__init__.py import prefix"
  fi
  if grep -q '"pix2pix\.models\."' "$CYTRAN/models/__init__.py"; then
    sed_inplace 's#"pix2pix\.models\." + model_name#"models." + model_name#' "$CYTRAN/models/__init__.py"
    echo "  patched models/__init__.py import prefix"
  fi
  # Our loader. cytran_vindr_dataset.py at the repo root stays the source of
  # truth; setup copies it into place so the path matches CyTran's loader lookup.
  cp "$HERE/cytran_vindr_dataset.py" "$CYTRAN/data/vindr_dataset.py"
  echo "  installed data/vindr_dataset.py (--dataset_mode vindr)"

  echo
  echo "  deps: pip install -r $HERE/requirements_cytran.txt  (in a dedicated env)"
}

do_prep() {
  log prep "our NIfTI -> pix2pix AB-PNG slices at $DATAROOT"
  resolve_split
  "$PY" "$HERE/prep_benchmark_data.py" --split "$SPLIT" \
      --format pix2pix --out "$DATAROOT" --size "$SIZE"
}

do_train() {
  [ -f "$CYTRAN/data/vindr_dataset.py" ] || { echo "ERROR: run setup first" >&2; exit 1; }
  log train "CyTran A=NCCT -> B=CECT ($NITER + $NITER_DECAY epochs)"
  cd "$CYTRAN"
  "$PY" train.py --dataroot "$DATAROOT" --name "$NAME" \
      --gpu_ids "$GPU" --device "$DEVICE" \
      --model cytran --dataset_mode vindr --direction AtoB --phase train \
      --input_nc 1 --output_nc 1 \
      --img_size "$SIZE" --load_size "$SIZE" --crop_size "$SIZE" \
      --ngf_cytran "$NGF_CYTRAN" --n_downsampling "$N_DOWNSAMPLING" \
      --depth "$DEPTH" --heads "$HEADS" \
      --batch_size "$BATCH" --lr "$LR" \
      --n_epochs "$NITER" --n_epochs_decay "$NITER_DECAY" \
      --save_epoch_freq "$SAVE_EPOCH_FREQ" --checkpoints_dir checkpoints \
      --display_id 0 --no_html
}

do_infer() {
  local ckpt="$CYTRAN/checkpoints/$NAME/${CKPT_EPOCH}_net_G_A.pth"
  if [ ! -f "$ckpt" ]; then
    echo "ERROR: generator weights missing: $ckpt" >&2
    echo "  train first, or pick an epoch that exists:" >&2
    ls "$CYTRAN/checkpoints/$NAME/" 2>/dev/null | grep '_net_G_A.pth' >&2 || true
    echo "  (train.py only writes 'latest' every SAVE_EPOCH_FREQ=$SAVE_EPOCH_FREQ epochs)" >&2
    exit 1
  fi
  log infer "G_A over the test split -> *_fake_B.npy"
  "$PY" "$HERE/infer_cytran.py" --cytran "$CYTRAN" \
      --index "$DATAROOT/slice_index.csv" --checkpoint "$ckpt" \
      --out "$SLICES" --device "$DEVICE" --batch_size 8 \
      --input_nc 1 --ngf_cytran "$NGF_CYTRAN" \
      --n_downsampling "$N_DOWNSAMPLING" --depth "$DEPTH" --heads "$HEADS"
}

do_reassemble() {
  log reassemble "slices -> per-case CECT NIfTI (HU) + manifest.csv"
  # infer_cytran.py saves the raw generator output, fed in [-1,1] -> neg1_1.
  # Explicit anchored pattern instead of the default heuristic: our case_ids are
  # dotted DICOM UIDs, and a greedy prefix + fixed 4-digit z + literal suffix
  # cannot mis-split them (the default's non-greedy .+? would if a case_id ever
  # contained an underscore).
  "$PY" "$HERE/reassemble_nifti.py" \
      --index "$DATAROOT/slice_index.csv" \
      --slices_dir "$SLICES" \
      --pattern '^(?P<case>.+)_(?P<z>[0-9]{4})_fake_B\.npy$' \
      --in_range neg1_1 --out "$OUT_NIFTI"
  echo
  echo "  manifest -> $OUT_NIFTI/manifest.csv"
  echo "  score it: cd ../synthetic_CECT && python benchmark.py \\"
  echo "      --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \\"
  echo "      --manifest cytran=$OUT_NIFTI/manifest.csv --baseline ours --out analysis/benchmark"
}

# ---- dispatch ---------------------------------------------------------------
stage="${1:-all}"
case "$stage" in
  setup)      do_setup ;;
  prep)       do_prep ;;
  train)      do_train ;;
  infer)      do_infer ;;
  reassemble) do_reassemble ;;
  all)        do_setup; do_prep; do_train; do_infer; do_reassemble ;;
  *) echo "usage: $0 [setup|prep|train|infer|reassemble|all]" >&2; exit 2 ;;
esac
log "$stage" "done"

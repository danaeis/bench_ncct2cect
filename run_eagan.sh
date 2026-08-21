#!/usr/bin/env bash
# =============================================================================
# run_eagan.sh — end-to-end Ea-GANs NCCT->CECT run on OUR VinDr data.
#
# Fifth benchmark module. Same contract as the others: train on the shared
# split, emit per-case synthetic CECT NIfTI (HU, source grid) + manifest.csv
# for ../synthetic_CECT/benchmark.py.
#
#   SPLIT=../synthetic_CECT/splits/split.json GPU=0 ./run_eagan.sh all
#   SPLIT=../synthetic_CECT/splits/split.json GPU=0 MODEL=dea_gan ./run_eagan.sh all
#
# Stages (run one, or `all`):
#   setup       verify the vendored Ea-GANs tree
#   prep        our NIfTI -> pix2pix AB-PNG slices (reuses ResViT's if present)
#   train       gEa-GAN or dEa-GAN (MODEL=gea_gan|dea_gan), A=NCCT -> B=CECT
#   test        netG over the test slices -> <case>_<zzzz>_fake_B.png
#   reassemble  slices -> per-case CECT NIfTI (HU, source grid) + manifest.csv
#   all         every stage above, in order
#
# See RUN_EAGAN.md for why this is not a straight re-vendor of upstream
# (github.com/by-lab/Ea-GANs): upstream is natively volumetric (Conv3d, 3-D
# SimpleITK-loaded patches, whole-volume .nii.gz output) — this repo's Ea-GANs/
# is a 2-D port (Conv2d, the same pix2pix AB-PNG slices every other model here
# trains on) so it is comparable to ResViT/CyTran/CycleGAN on the same split
# and the same per-slice metrics, at the cost of not being bit-for-bit the
# upstream code. models/networks.py's docstring has the exact conversion.
# =============================================================================
set -euo pipefail

# ---- config (override via env) ----------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EAGAN="$HERE/Ea-GANs"

MODEL="${MODEL:-gea_gan}"        # gea_gan (edge loss on G only) | dea_gan (G+D)
case "$MODEL" in gea_gan|dea_gan) ;; *) echo "ERROR: MODEL must be gea_gan or dea_gan, got: $MODEL" >&2; exit 2 ;; esac

SPLIT="${SPLIT:-}"                                              # shared case split
# Reuses ResViT's prepped slices by default — identical pix2pix AB-PNG format,
# same trick SwinUNETR/TransUNet already use (RUN_BACKBONES.md). prep is a
# no-op when they already exist.
DATAROOT="${DATAROOT:-$HERE/ResViT/datasets/vindr}"
SIZE="${SIZE:-256}"
MIN_TISSUE_FRAC="${MIN_TISSUE_FRAC:-0.0}"
GPU="${GPU:-0}"                  # gpu id; -1 = cpu
NAME="${NAME:-vindr_${MODEL}}"
NITER="${NITER:-25}"             # --niter        (epochs at full lr)
NITER_DECAY="${NITER_DECAY:-25}" # --niter_decay  (epochs of linear decay)
BATCH="${BATCH:-4}"
LR="${LR:-0.0002}"               # upstream's default
NORM="${NORM:-batch}"            # upstream's default; kept for fidelity to the paper
LAMBDA_A="${LAMBDA_A:-100}"      # weight on the L1 reconstruction term (upstream's own scripts use 300; 100 matches ResViT/CyTran's convention on this data — override if you want upstream's literal value)
LAMBDA_SOBEL="${LAMBDA_SOBEL:-100}"
SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-5}"
CKPT_EPOCH="${CKPT_EPOCH:-latest}"
GPU_IDS="$GPU"
OUT_NIFTI="${OUT_NIFTI:-$HERE/results/vindr_${MODEL}_nifti}"
SLICES="$EAGAN/results/$NAME/test_$CKPT_EPOCH/images"

# Resume an interrupted run instead of restarting at epoch 1 and overwriting the
# checkpoints. Unlike CycleGAN/ResViT, upstream train.py never supported this at
# all (the epoch loop always started at 1) — see options/train_options.py's
# --epoch_count and train.py, added here.
RESUME="${RESUME:-1}"
EPOCH_COUNT="${EPOCH_COUNT:-}"

PY="${PYTHON:-python3}"

if [ "${DISABLE_CUDNN:-0}" = 1 ]; then RUNPY=("$PY" "$HERE/nocudnn.py"); else RUNPY=("$PY"); fi

log() { printf '\n\033[1;36m[run_eagan:%s]\033[0m %s\n' "$1" "$2"; }

require_cuda() {
  [ "${ALLOW_CPU:-0}" = 1 ] && return 0
  [ "$GPU" = "-1" ] && return 0
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$HERE/preflight_gpu.py" || {
    echo "  (preflight failed — refusing to start; ALLOW_CPU=1 overrides)" >&2; exit 1; }
}

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

# Newest <epoch>_net_G.pth in a checkpoints dir, or empty if there are none.
latest_numbered_epoch() {
  ls "$1"/[0-9]*_net_G.pth 2>/dev/null \
    | sed 's#.*/##; s#_net_G\.pth$##' \
    | grep -E '^[0-9]+$' | sort -n | tail -1
}

# Populate $RESUME_ARGS. Same caveat as run_cyclegan.sh's set_resume_args: the
# restart point comes from the newest NUMBERED checkpoint (every
# SAVE_EPOCH_FREQ epochs), while latest_net_*.pth is refreshed every
# --save_latest_freq iterations, so the reloaded weights can be slightly ahead
# of the epoch the lr/sobel schedule resumes at. The lr schedule itself also
# restarts from --lr on resume (old_lr is not persisted) — a known imprecision,
# not bit-identical to an uninterrupted run, but safe.
RESUME_ARGS=()
set_resume_args() {
  RESUME_ARGS=()
  local dir="$1" n start
  [ "$RESUME" = 1 ] || return 0
  if [ ! -f "$dir/latest_net_G.pth" ]; then
    echo "  RESUME: no $dir/latest_net_G.pth — starting from scratch"
    return 0
  fi
  n="$(latest_numbered_epoch "$dir")"
  start="${EPOCH_COUNT:-$(( ${n:-0} + 1 ))}"
  RESUME_ARGS=(--continue_train --which_epoch latest --epoch_count "$start")
  echo "  RESUME: continuing from $dir/latest_net_G.pth at epoch $start"
}

# ---- stages -----------------------------------------------------------------
do_setup() {
  log setup "check Ea-GANs tree"
  [ -f "$EAGAN/train.py" ] || { echo "ERROR: Ea-GANs/ is missing or empty at $EAGAN" >&2; exit 1; }
  [ -d "$EAGAN/data" ] || { echo "ERROR: Ea-GANs/data/ missing — this is the 2-D port, not upstream by-lab/Ea-GANs as-is" >&2; exit 1; }
  echo "  present: $EAGAN"
  echo "  deps: pip install -r $HERE/requirements_eagan.txt  (in a dedicated env)"
}

do_prep() {
  resolve_split
  if [ "${REPREP:-0}" != 1 ] && [ -d "$DATAROOT/train" ] && [ -f "$DATAROOT/slice_index.csv" ]; then
    log prep "reusing existing slices at $DATAROOT (REPREP=1 to force)"
    echo "  train slices: $(ls "$DATAROOT/train" | wc -l | tr -d ' ')"
    return
  fi
  log prep "our NIfTI -> pix2pix AB-PNG slices at $DATAROOT"
  "$PY" "$HERE/prep_benchmark_data.py" --split "$SPLIT" \
      --format pix2pix --out "$DATAROOT" --size "$SIZE" \
      --min_tissue_frac "$MIN_TISSUE_FRAC"
}

do_train() {
  [ -d "$DATAROOT/train" ] || { echo "ERROR: $DATAROOT/train missing (run prep first)" >&2; exit 1; }
  require_cuda
  log train "$MODEL A=NCCT -> B=CECT ($NITER + $NITER_DECAY epochs)"
  set_resume_args "$EAGAN/checkpoints/$NAME"
  cd "$EAGAN"
  "${RUNPY[@]}" train.py --dataroot "$DATAROOT" --name "$NAME" \
      ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
      --model "$MODEL" --dataset_mode aligned --which_direction AtoB \
      --input_nc 1 --output_nc 1 --loadSize "$SIZE" --fineSize "$SIZE" \
      --which_model_netG unet_128 --norm "$NORM" --gpu_ids "$GPU_IDS" \
      --lambda_A "$LAMBDA_A" --lambda_sobel "$LAMBDA_SOBEL" \
      --batchSize "$BATCH" --lr "$LR" \
      --niter "$NITER" --niter_decay "$NITER_DECAY" \
      --save_epoch_freq "$SAVE_EPOCH_FREQ" --checkpoints_dir checkpoints \
      --display_id 0 --no_html --no_flip
}

do_test() {
  local w="$EAGAN/checkpoints/$NAME/${CKPT_EPOCH}_net_G.pth"
  if [ ! -f "$w" ]; then
    echo "ERROR: generator weights missing: $w" >&2
    echo "  train first, or pick an epoch that exists:" >&2
    ls "$EAGAN/checkpoints/$NAME/" 2>/dev/null | grep '_net_G.pth' >&2 || true
    exit 1
  fi
  require_cuda
  log test "$MODEL netG over the test split -> *_fake_B.png"
  cd "$EAGAN"
  "${RUNPY[@]}" test.py --dataroot "$DATAROOT" --name "$NAME" \
      --model "$MODEL" --dataset_mode aligned --which_direction AtoB \
      --input_nc 1 --output_nc 1 --loadSize "$SIZE" --fineSize "$SIZE" \
      --which_model_netG unet_128 --norm "$NORM" --gpu_ids "$GPU_IDS" \
      --which_epoch "$CKPT_EPOCH" --how_many 1000000 \
      --results_dir results
}

do_reassemble() {
  log reassemble "slices -> per-case CECT NIfTI (HU) + manifest.csv"
  # test.py's save_images writes '<case>_<zzzz>_<label>.png' via tensor2im
  # ([0,255] display range, same convention as ResViT).
  "$PY" "$HERE/reassemble_nifti.py" \
      --index "$DATAROOT/slice_index.csv" \
      --slices_dir "$SLICES" \
      --slice_suffix _fake_B --in_range 0_255 --out "$OUT_NIFTI"
  echo
  echo "  manifest -> $OUT_NIFTI/manifest.csv"
  echo "  score it: cd ../synthetic_CECT && python benchmark.py \\"
  echo "      --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \\"
  echo "      --manifest ${MODEL}=$OUT_NIFTI/manifest.csv --baseline ours --out analysis/bench_ncct2cect"
}

# ---- dispatch ---------------------------------------------------------------
stage="${1:-all}"
case "$stage" in
  setup)      do_setup ;;
  prep)       do_prep ;;
  train)      do_train ;;
  test)       do_test ;;
  reassemble) do_reassemble ;;
  all)        (do_setup); (do_prep); (do_train); (do_test); (do_reassemble) ;;
  *) echo "usage: $0 [setup|prep|train|test|reassemble|all]" >&2; exit 2 ;;
esac
log "$stage" "done"

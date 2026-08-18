#!/usr/bin/env bash
# =============================================================================
# run_resvit.sh — end-to-end ResViT NCCT->CECT run on OUR VinDr data.
#
# Prepped to launch on a GPU server (CUDA >= 11.2). Nothing here needs editing;
# override any variable from the environment, e.g.:
#
#   SPLIT=../synthetic_CECT/splits/split.json GPU=0 ./run_resvit.sh all
#
# Stages (run one, or `all`):
#   setup      verify the vendored ResViT tree, apply compat shims,
#              fetch the pretrained ViT checkpoint
#   prep       our NIfTI -> pix2pix AB-PNG slices (train/val/test) on the split
#   pretrain   stage 1: pretrain the ART/res_cnn generator (no transformer)
#   finetune   stage 2: fine-tune the full ResViT from the pretrained weights
#   test       inference on the test split -> per-slice *_fake_B.png
#   reassemble slices -> per-case CECT NIfTI (HU, source grid) + manifest.csv
#   all        every stage above, in order
#
# The pipeline follows ResViT/README's two-stage recipe (pretrain res_cnn ->
# fine-tune resvit) for a one-to-one task (input_nc=1, output_nc=1, AtoB =
# NCCT->CECT). The BENCHMARK.md snippet omitted the pretrain stage; this does it.
# =============================================================================
set -euo pipefail

# ---- config (override via env) ----------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESVIT="$HERE/ResViT"

SPLIT="${SPLIT:-}"                                              # shared case split
DATAROOT="${DATAROOT:-$RESVIT/datasets/vindr}"                  # prepped slices
SIZE="${SIZE:-256}"
# Drop train/val slices with < this fraction of >0.1 voxels (0 = keep every
# slice, the default). Test is never filtered, so reconstruction stays complete.
MIN_TISSUE_FRAC="${MIN_TISSUE_FRAC:-0.0}"
GPU="${GPU:-0}"                                                 # gpu id; -1 = cpu
NAME="${NAME:-vindr_resvit}"                                    # finetune exp name
PRE_NAME="${PRE_NAME:-vindr_resvit_pretrain}"                   # pretrain exp name
NITER="${NITER:-25}"            # epochs at full lr  (pretrain uses 2x these)
NITER_DECAY="${NITER_DECAY:-25}" # epochs of lr decay
BATCH="${BATCH:-1}"
# train.py only writes a numbered checkpoint AND a validation line on epochs
# divisible by this (train.py:92). A run shorter than one interval therefore ends
# with no numbered checkpoint and an empty log.txt — set SAVE_EPOCH_FREQ=1 for
# smoke tests so you can tell a short run from a crashed one.
SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-5}"
# Resume an interrupted stage instead of silently restarting at epoch 1 and
# overwriting the checkpoints. Same reasoning as SynDiff's RESUME: these runs are
# far too long to redo because a box rebooted. EPOCH_COUNT overrides the epoch
# the schedule restarts from (default: one past the newest numbered checkpoint).
RESUME="${RESUME:-1}"
EPOCH_COUNT="${EPOCH_COUNT:-}"
OUT_NIFTI="${OUT_NIFTI:-$RESVIT/results/vindr_nifti}"

VIT_DIR="$RESVIT/model/vit_checkpoint/imagenet21k"
VIT_CKPT="$VIT_DIR/R50+ViT-B_16.npz"   # exact name models/transformer_configs.py loads
VIT_URL="https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz"

PY="${PYTHON:-python3}"

# DISABLE_CUDNN=1 routes the upstream entrypoints through nocudnn.py, which turns
# cuDNN off before the first conv. For a broken cuDNN runtime on an otherwise
# supported GPU (see preflight_gpu.py); costs speed, needs no reinstall.
if [ "${DISABLE_CUDNN:-0}" = 1 ]; then RUNPY=("$PY" "$HERE/nocudnn.py"); else RUNPY=("$PY"); fi
GPU_IDS="$GPU"

# Never route loopback through the site proxy: it answers CONNECT with 403 and the
# client stalls on retries. (Outbound URLs, e.g. the ViT checkpoint, still proxy.)
export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

# TensorFlow, if anything in the env drags it in (MONAI pulls TensorBoard, which
# prefers real TF when installed), grabs essentially ALL GPU memory on first use.
# Harmless no-op when TF is absent; prevents a starved trainer when it is not.
export TF_FORCE_GPU_ALLOW_GROWTH=true


log() { printf '\n\033[1;36m[run_resvit:%s]\033[0m %s\n' "$1" "$2"; }

# Fail fast rather than training on CPU, or dying at the first conv on a GPU this
# torch/cuDNN cannot drive. See preflight_gpu.py. GPU=-1 or ALLOW_CPU=1 to skip.
require_cuda() {
  [ "${ALLOW_CPU:-0}" = 1 ] && return 0
  [ "$GPU" = "-1" ] && return 0
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$HERE/preflight_gpu.py" || {
    echo "  (preflight failed — refusing to start; ALLOW_CPU=1 overrides)" >&2; exit 1; }
}

# `sed -i` takes a mandatory suffix on BSD/macOS and none on GNU; write through a
# temp file so this works on either. Fails loudly rather than leaving the source
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

# ResViT/ is vendored, so these patches are committed and this is normally a
# no-op. Kept, and called from every stage that runs ResViT code, so a fresh
# checkout of upstream ResViT dropped in here still works. Idempotent.
apply_shims() {
  # 1) skimage>=0.18 removed compare_psnr -> use peak_signal_noise_ratio.
  if grep -q "from skimage.measure import compare_psnr as psnr" "$RESVIT/train.py"; then
    sed_inplace 's#from skimage.measure import compare_psnr as psnr#from skimage.metrics import peak_signal_noise_ratio as psnr#' "$RESVIT/train.py"
    echo "  shim: train.py psnr import"
  fi
  # 2) scipy>=1.3 removed scipy.misc.imresize; it is only used when aspect_ratio
  #    != 1.0 (never in our path), but the top-level import must not crash.
  # NB: delimiter is '|', not '#' — the replacement text contains a '#' comment,
  # which would otherwise terminate the s/// early and be parsed as flags.
  if grep -q "from scipy.misc import imresize" "$RESVIT/util/visualizer.py"; then
    sed_inplace 's|from scipy.misc import imresize|imresize = lambda im, size, interp=None: im  # shim: aspect_ratio stays 1.0|' "$RESVIT/util/visualizer.py"
    echo "  shim: visualizer.py imresize import"
  fi
}

# Newest <epoch>_net_G.pth in a checkpoints dir, or empty if there are none.
latest_numbered_epoch() {
  ls "$1"/[0-9]*_net_G.pth 2>/dev/null \
    | sed 's#.*/##; s#_net_G\.pth$##' \
    | grep -E '^[0-9]+$' | sort -n | tail -1
}

# Populate $RESUME_ARGS for an experiment dir: the flags that make train.py pick
# up where it stopped, or nothing at all if there is no checkpoint to resume.
#
# ResViT records no "current epoch" anywhere, so the restart point is derived
# from the newest NUMBERED checkpoint (written every --save_epoch_freq epochs).
# `latest_net_G.pth` is written more often than that (every --save_latest_freq
# iterations), so the weights we reload can be slightly AHEAD of the epoch we
# tell the lr schedule to resume at. That errs on the safe side — it replays at
# most SAVE_EPOCH_FREQ epochs of schedule — but it does mean the lr curve of a
# resumed run is not bit-identical to an uninterrupted one. Pin EPOCH_COUNT
# yourself if you need it exact.
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
  log setup "check ResViT tree"
  [ -f "$RESVIT/train.py" ] || { echo "ERROR: ResViT/ is missing or empty at $RESVIT" >&2; exit 1; }
  echo "  present: $RESVIT"

  log setup "apply compat shims (idempotent)"
  apply_shims

  log setup "fetch pretrained ViT checkpoint"
  if [ -f "$VIT_CKPT" ]; then
    echo "  present: $VIT_CKPT"
  else
    mkdir -p "$VIT_DIR"
    # curl handles the literal '+' via %2B; -L follows redirects, -f fails on 4xx
    curl -fL "$VIT_URL" -o "$VIT_CKPT"
    echo "  downloaded -> $VIT_CKPT"
  fi

  echo
  echo "  deps: pip install -r $HERE/requirements_resvit.txt  (in a dedicated env)"
}

do_prep() {
  log prep "our NIfTI -> pix2pix AB-PNG slices at $DATAROOT"
  resolve_split
  "$PY" "$HERE/prep_benchmark_data.py" --split "$SPLIT" \
      --format pix2pix --out "$DATAROOT" --size "$SIZE" \
      --min_tissue_frac "$MIN_TISSUE_FRAC"
}

do_pretrain() {
  require_cuda
  log pretrain "stage 1: res_cnn generator ($((NITER*2)) + $((NITER_DECAY*2)) epochs)"
  apply_shims
  set_resume_args "$RESVIT/checkpoints/$PRE_NAME"
  cd "$RESVIT"
  "${RUNPY[@]}" train.py --dataroot "$DATAROOT" --name "$PRE_NAME" --gpu_ids "$GPU_IDS" \
      ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
      --model resvit_one --which_model_netG res_cnn --which_direction AtoB \
      --lambda_A 100 --dataset_mode aligned --norm batch --pool_size 0 \
      --input_nc 1 --output_nc 1 --loadSize "$SIZE" --fineSize "$SIZE" \
      --niter $((NITER*2)) --niter_decay $((NITER_DECAY*2)) --save_epoch_freq "$SAVE_EPOCH_FREQ" \
      --checkpoints_dir checkpoints/ --display_id 0 --lr 0.0002 --batchSize "$BATCH"
}

do_finetune() {
  local pre_weights="$RESVIT/checkpoints/$PRE_NAME/latest_net_G.pth"
  [ -f "$pre_weights" ] || { echo "ERROR: pretrain weights missing: $pre_weights (run pretrain first)" >&2; exit 1; }
  require_cuda
  log finetune "stage 2: full ResViT from $pre_weights"
  apply_shims
  # NB: on a resume the --pre_trained_* flags below are harmless — resvit_one
  # loads the pretrained ART/transformer weights first, then --continue_train
  # overwrites netG/netD with the fine-tune checkpoint.
  set_resume_args "$RESVIT/checkpoints/$NAME"
  cd "$RESVIT"
  "${RUNPY[@]}" train.py --dataroot "$DATAROOT" --name "$NAME" --gpu_ids "$GPU_IDS" \
      ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
      --model resvit_one --which_model_netG resvit --which_direction AtoB \
      --lambda_A 100 --dataset_mode aligned --norm batch --pool_size 0 \
      --input_nc 1 --output_nc 1 --loadSize "$SIZE" --fineSize "$SIZE" \
      --niter "$NITER" --niter_decay "$NITER_DECAY" --save_epoch_freq "$SAVE_EPOCH_FREQ" \
      --checkpoints_dir checkpoints/ --display_id 0 \
      --pre_trained_transformer 1 --pre_trained_resnet 1 \
      --pre_trained_path "checkpoints/$PRE_NAME/latest_net_G.pth" \
      --lr 0.001 --batchSize "$BATCH"
}

do_test() {
  require_cuda
  log test "inference on the test split -> *_fake_B.png"
  apply_shims
  cd "$RESVIT"
  "${RUNPY[@]}" test.py --dataroot "$DATAROOT" --name "$NAME" --gpu_ids "$GPU_IDS" \
      --model resvit_one --which_model_netG resvit --dataset_mode aligned \
      --norm batch --phase test --input_nc 1 --output_nc 1 \
      --loadSize "$SIZE" --fineSize "$SIZE" --how_many 1000000 --serial_batches \
      --results_dir results/ --checkpoints_dir checkpoints/ --which_epoch latest \
      --display_id 0
}

do_reassemble() {
  log reassemble "slices -> per-case CECT NIfTI (HU) + manifest.csv"
  # tensor2im saves display PNGs in [0,255] -> in_range 0_255.
  "$PY" "$HERE/reassemble_nifti.py" \
      --index "$DATAROOT/slice_index.csv" \
      --slices_dir "$RESVIT/results/$NAME/test_latest/images" \
      --slice_suffix _fake_B --in_range 0_255 --out "$OUT_NIFTI"
  echo
  echo "  manifest -> $OUT_NIFTI/manifest.csv"
  echo "  score it: cd ../synthetic_CECT && python benchmark.py \\"
  echo "      --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \\"
  echo "      --manifest resvit=$OUT_NIFTI/manifest.csv --baseline ours --out analysis/benchmark"
}

# ---- dispatch ---------------------------------------------------------------
stage="${1:-all}"
case "$stage" in
  setup)      do_setup ;;
  prep)       do_prep ;;
  pretrain)   do_pretrain ;;
  finetune)   do_finetune ;;
  test)       do_test ;;
  reassemble) do_reassemble ;;
# Each stage runs in its own SUBSHELL. Stages that train a vendored repo `cd`
# into it, and that CWD used to leak into the stages that followed — which is how
# `reassemble` ended up resolving the split's relative source paths against
# <repo>/<vendor>/ and dying with "No such file or no access".
  all)        (do_setup); (do_prep); (do_pretrain); (do_finetune); (do_test); (do_reassemble) ;;
  *) echo "usage: $0 [setup|prep|pretrain|finetune|test|reassemble|all]" >&2; exit 2 ;;
esac
log "$stage" "done"

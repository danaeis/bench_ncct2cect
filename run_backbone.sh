#!/usr/bin/env bash
# =============================================================================
# run_backbone.sh — shared runner for the two repurposed segmentation backbones,
# SwinUNETR and TransUNet, on OUR VinDr NCCT->CECT data.
#
# You normally invoke it through one of the two thin wrappers, which just pin
# $ARCH and the default experiment name:
#
#   SPLIT=../synthetic_CECT/splits/split.json GPU=0 ./run_swinunetr.sh all
#   SPLIT=../synthetic_CECT/splits/split.json GPU=0 ./run_transunet.sh all
#
# Stages (run one, or `all`):
#   setup       verify the vendored tree + deps; fetch the ViT init (TransUNet)
#   prep        our NIfTI -> pix2pix AB-PNG slices (SHARED with ResViT, see below)
#   train       train_backbone.py — ResViT's recipe (PatchGAN + 100*L1)
#   test        infer_backbone.py over the test split -> *_fake_B.npy
#   reassemble  slices -> per-case CECT NIfTI (HU, source grid) + manifest.csv
#   all         every stage above, in order
#
# Why these two are not "just another adapter":
#
#   Every other model in this benchmark ships a training pipeline; we only
#   convert data for it. SwinUNETR and TransUNet ship an *architecture* plus a
#   segmentation trainer (Dice/CE on BTCV/Synapse/BraTS) that has no bearing on
#   image synthesis. So for these we keep the upstream networks byte-for-byte
#   and supply the training loop ourselves, deliberately copying ResViT's
#   objective so the comparison isolates architecture. See backbone_common.py.
#
#   DATAROOT defaults to ResViT's prepped slices. The pix2pix AB-PNG layout is
#   byte-identical to what these need, and a second copy of every axial slice is
#   pure waste. `prep` is therefore a no-op when the dir is already populated;
#   force it with REPREP=1, or point DATAROOT somewhere else.
# =============================================================================
set -euo pipefail

# ---- config (override via env) ----------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCH="${ARCH:?ARCH must be set to swinunetr or transunet (use run_swinunetr.sh / run_transunet.sh)}"
case "$ARCH" in
  swinunetr|transunet) ;;
  *) echo "ERROR: ARCH must be 'swinunetr' or 'transunet', got '$ARCH'" >&2; exit 2 ;;
esac

SPLIT="${SPLIT:-}"                                              # shared case split
# Shared with ResViT on purpose — same format, one copy on disk.
DATAROOT="${DATAROOT:-$HERE/ResViT/datasets/vindr}"
REPREP="${REPREP:-0}"           # 1 = re-run prep even if DATAROOT looks populated
SIZE="${SIZE:-256}"
MIN_TISSUE_FRAC="${MIN_TISSUE_FRAC:-0.0}"
GPU="${GPU:-0}"
NAME="${NAME:-vindr_$ARCH}"
CKPT_DIR="${CKPT_DIR:-$HERE/checkpoints}"
NITER="${NITER:-25}"            # epochs at full lr
NITER_DECAY="${NITER_DECAY:-25}" # epochs of linear lr decay
# These are 2-D nets on 256x256 slices, so a real batch fits and helps the
# BatchNorm in the discriminator. Drop to 2 (TransUNet) if the GPU is small.
BATCH="${BATCH:-4}"
WORKERS="${WORKERS:-4}"
LR="${LR:-2e-4}"
LAMBDA_L1="${LAMBDA_L1:-100}"   # ResViT's --lambda_A
LAMBDA_ADV="${LAMBDA_ADV:-1.0}" # 0 -> pure-L1 ablation, no discriminator
SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-5}"
FEATURE_SIZE="${FEATURE_SIZE:-48}"   # SwinUNETR width; must be a multiple of 12
# 2.5-D context: odd number of adjacent axial slices fed as input channels,
# predicting the centre slice. 1 = plain 2-D. 5/7/9/11 mirror the slices5_k2 /
# slices11_k5 runs in ../synthetic_CECT, so the numbers stay comparable to your
# own model. SwinUNETR only — see backbone_common.py for why TransUNet refuses.
N_SLICES="${N_SLICES:-1}"
WHICH_EPOCH="${WHICH_EPOCH:-latest}"
CONTINUE="${CONTINUE:-0}"       # 1 = resume from latest_net_G.pth
OUT_NIFTI="${OUT_NIFTI:-$HERE/results/${NAME}_nifti}"
SLICES="${SLICES:-$CKPT_DIR/$NAME/test_slices}"

# TransUNet's R50-ViT-B_16 wants exactly the checkpoint ResViT already fetches.
VIT_DIR="$HERE/ResViT/model/vit_checkpoint/imagenet21k"
VIT_CKPT="${VIT_CKPT:-$VIT_DIR/R50+ViT-B_16.npz}"
VIT_URL="https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz"
# VIT_INIT=0 trains TransUNet from scratch — the like-for-like comparison against
# SwinUNETR, which has no 2-D pretrained weights available at all (see setup).
VIT_INIT="${VIT_INIT:-1}"

PY="${PYTHON:-python3}"

export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

# TensorFlow, if anything in the env drags it in (MONAI pulls TensorBoard, which
# prefers real TF when installed), grabs essentially ALL GPU memory on first use.
# Harmless no-op when TF is absent; prevents a starved trainer when it is not.
export TF_FORCE_GPU_ALLOW_GROWTH=true


log() { printf '\n\033[1;36m[run_%s:%s]\033[0m %s\n' "$ARCH" "$1" "$2"; }

# Fail loudly rather than training on CPU for days. A torch built for a newer
# CUDA than the driver supports still imports fine and silently reports
# is_available()==False, which is how a CycleGAN run burned a day printing
# "Initialized with device cpu". ALLOW_CPU=1 to override deliberately.
require_cuda() {
  [ "${ALLOW_CPU:-0}" = 1 ] && return 0
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

# ---- stages -----------------------------------------------------------------
do_setup() {
  log setup "check tree + deps"
  if [ "$ARCH" = transunet ]; then
    [ -f "$HERE/TransUNet/networks/vit_seg_modeling.py" ] \
      || { echo "ERROR: TransUNet/ is missing or empty at $HERE/TransUNet" >&2; exit 1; }
    echo "  present: $HERE/TransUNet"
  fi
  for m in torch numpy PIL nibabel; do
    "$PY" -c "import $m" 2>/dev/null && echo "  ok: $m" || {
      echo "  MISSING: $m   (pip install -r $HERE/requirements_backbone.txt)"; exit 1; }
  done
  if [ "$ARCH" = swinunetr ]; then
    "$PY" -c 'import monai' 2>/dev/null && echo "  ok: monai ($("$PY" -c 'import monai;print(monai.__version__)'))" || {
      echo "  MISSING: monai   (pip install -r $HERE/requirements_backbone.txt)"; exit 1; }
  else
    "$PY" -c 'import ml_collections, scipy' 2>/dev/null && echo "  ok: ml_collections, scipy" || {
      echo "  MISSING: ml_collections/scipy   (pip install -r $HERE/requirements_backbone.txt)"; exit 1; }
  fi

  if [ "$ARCH" = transunet ] && [ "$VIT_INIT" != 1 ]; then
    echo
    echo "  VIT_INIT=0 — TransUNet will train from scratch (no ImageNet-21k init)."
  elif [ "$ARCH" = transunet ]; then
    log setup "fetch the ImageNet-21k ViT init"
    if [ -f "$VIT_CKPT" ]; then
      echo "  present: $VIT_CKPT"
    else
      mkdir -p "$(dirname "$VIT_CKPT")"
      # Same ~400 MB file run_resvit.sh downloads; the literal '+' is %2B.
      curl -fL "$VIT_URL" -o "$VIT_CKPT"
      echo "  downloaded -> $VIT_CKPT"
    fi
  else
    echo
    echo "  NB: SwinUNETR trains from scratch here. The published self-supervised"
    echo "      SwinViT weights (model_swinvit.pt) are 3-D only and cannot load"
    echo "      into a spatial_dims=2 model. Report that alongside its numbers."
  fi
}

do_prep() {
  resolve_split
  if [ "$REPREP" != 1 ] && [ -d "$DATAROOT/train" ] && [ -f "$DATAROOT/slice_index.csv" ]; then
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
  log train "$ARCH ($NITER + $NITER_DECAY epochs, lambda_adv=$LAMBDA_ADV, n_slices=$N_SLICES)"
  local extra=()
  [ "$CONTINUE" = 1 ] && extra+=(--continue_train)
  if [ "$ARCH" = transunet ]; then
    extra+=(--transunet_dir "$HERE/TransUNet")
    # An empty --vit_ckpt is how train_backbone.py is told to skip the init.
    if [ "$VIT_INIT" = 1 ]; then extra+=(--vit_ckpt "$VIT_CKPT")
    else extra+=(--vit_ckpt ""); fi
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$HERE/train_backbone.py" \
      --arch "$ARCH" --dataroot "$DATAROOT" --name "$NAME" \
      --checkpoints_dir "$CKPT_DIR" --size "$SIZE" \
      --batch "$BATCH" --workers "$WORKERS" \
      --niter "$NITER" --niter_decay "$NITER_DECAY" --lr "$LR" \
      --lambda_l1 "$LAMBDA_L1" --lambda_adv "$LAMBDA_ADV" \
      --save_epoch_freq "$SAVE_EPOCH_FREQ" --feature_size "$FEATURE_SIZE" \
      --n_slices "$N_SLICES" \
      ${extra[@]+"${extra[@]}"}
      # ^ the `+` form is required: under `set -u`, bash 3.2 (macOS) treats a
      #   bare "${extra[@]}" on an empty array as an unbound variable.
}

do_test() {
  local w="$CKPT_DIR/$NAME/${WHICH_EPOCH}_net_G.pth"
  [ -f "$w" ] || { echo "ERROR: generator weights missing: $w (run train first)" >&2; exit 1; }
  require_cuda
  log test "inference on the test split -> *_fake_B.npy"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$HERE/infer_backbone.py" \
      --arch "$ARCH" --dataroot "$DATAROOT" --name "$NAME" \
      --checkpoints_dir "$CKPT_DIR" --which_epoch "$WHICH_EPOCH" \
      --out "$SLICES" --size "$SIZE" --batch "$BATCH" --workers "$WORKERS" \
      --feature_size "$FEATURE_SIZE" --n_slices "$N_SLICES" \
      --transunet_dir "$HERE/TransUNet"
}

do_reassemble() {
  log reassemble "slices -> per-case CECT NIfTI (HU) + manifest.csv"
  # infer_backbone.py saves the raw tanh output -> neg1_1. Anchored pattern for
  # the same reason as SynDiff/CyTran: our case_ids are dotted DICOM UIDs, and a
  # greedy prefix plus a fixed 4-digit z cannot mis-split them.
  "$PY" "$HERE/reassemble_nifti.py" \
      --index "$DATAROOT/slice_index.csv" \
      --slices_dir "$SLICES" \
      --pattern '^(?P<case>.+)_(?P<z>[0-9]{4})_fake_B\.npy$' \
      --in_range neg1_1 --out "$OUT_NIFTI"
  echo
  echo "  manifest -> $OUT_NIFTI/manifest.csv"
  echo "  score it: cd ../synthetic_CECT && python benchmark.py \\"
  echo "      --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \\"
  echo "      --manifest $ARCH=$OUT_NIFTI/manifest.csv --baseline ours --out analysis/benchmark"
}

# ---- dispatch ---------------------------------------------------------------
stage="${1:-all}"
case "$stage" in
  setup)      do_setup ;;
  prep)       do_prep ;;
  train)      do_train ;;
  test)       do_test ;;
  reassemble) do_reassemble ;;
# Each stage runs in its own SUBSHELL. Stages that train a vendored repo `cd`
# into it, and that CWD used to leak into the stages that followed — which is how
# `reassemble` ended up resolving the split's relative source paths against
# <repo>/<vendor>/ and dying with "No such file or no access".
  all)        (do_setup); (do_prep); (do_train); (do_test); (do_reassemble) ;;
  *) echo "usage: $0 [setup|prep|train|test|reassemble|all]" >&2; exit 2 ;;
esac
log "$stage" "done"

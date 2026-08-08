#!/usr/bin/env bash
# =============================================================================
# run_syndiff.sh — end-to-end SynDiff NCCT->CECT run on OUR VinDr data.
#
# Third benchmark module, after ResViT and CyTran. Same contract: train on the
# shared split, emit per-case synthetic CECT NIfTI (HU, source grid) +
# manifest.csv for ../synthetic_CECT/benchmark.py.
#
#   SPLIT=../synthetic_CECT/splits/split.json GPU=0 ./run_syndiff.sh all
#
# Stages (run one, or `all`):
#   setup       verify the vendored SynDiff tree, check the JIT-CUDA toolchain
#   prep        our NIfTI -> data_{train,val,test}_{NCCT,CECT}.mat on the split
#   train       SynDiff adversarial diffusion, contrast1=NCCT -> contrast2=CECT
#   infer       gen_diffusive_2 over the test slices -> per-slice *_fake_B.npy
#   reassemble  slices -> per-case CECT NIfTI (HU, source grid) + manifest.csv
#   all         every stage above, in order
#
# SynDiff differs from the other two in three ways that shape this script:
#   * data format is HDF5 .mat (`data_<phase>_<contrast>.mat`, key `data_fs`),
#     not PNG pairs -> prep runs with `--format mat`.
#   * train.py always goes through torch.distributed with the NCCL backend,
#     even for one GPU, so it needs a free MASTER port and a real CUDA device.
#   * backbones import utils/op, which JIT-compiles CUDA kernels at import via
#     torch.utils.cpp_extension.load -> needs ninja + nvcc + pythonX-dev.
#     `setup` preflights this so it fails here, not 40 minutes into training.
#
# Its test.py is unusable for us (hardcoded 256x152 IXI crop, per-slice max
# renormalisation, no case/z in the outputs) -> we use infer_syndiff.py.
# =============================================================================
set -euo pipefail

# ---- config (override via env) ----------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNDIFF="$HERE/SynDiff"

SPLIT="${SPLIT:-}"                                              # shared case split
DATAROOT="${DATAROOT:-$SYNDIFF/datasets/vindr}"                 # prepped .mat
SIZE="${SIZE:-256}"             # SynDiff pads/assumes 256; do not change lightly
GPU="${GPU:-0}"                 # gpu id; CPU is not supported (NCCL + JIT CUDA)
NAME="${NAME:-vindr_syndiff}"   # --exp
OUTPUT_PATH="${OUTPUT_PATH:-$SYNDIFF/results}"                  # holds <NAME>/
NUM_EPOCH="${NUM_EPOCH:-50}"
BATCH="${BATCH:-1}"
SAVE_CKPT_EVERY="${SAVE_CKPT_EVERY:-10}"   # infer can only load a saved epoch
CKPT_EPOCH="${CKPT_EPOCH:-}"               # default: newest saved epoch
PORT="${PORT:-6036}"                       # DDP MASTER_PORT
OUT_NIFTI="${OUT_NIFTI:-$SYNDIFF/results/vindr_nifti}"

# architecture — training and inference MUST agree on every one of these
NUM_CHANNELS="${NUM_CHANNELS:-2}"          # noise + source, both 1-channel
NUM_CHANNELS_DAE="${NUM_CHANNELS_DAE:-64}"
CH_MULT="${CH_MULT:-1 1 2 2 4 4}"
NUM_RES_BLOCKS="${NUM_RES_BLOCKS:-2}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-4}"
NGF="${NGF:-64}"
Z_EMB_DIM="${Z_EMB_DIM:-256}"
LR_D="${LR_D:-1e-4}"
LR_G="${LR_G:-1.6e-4}"

CONTRAST1="${CONTRAST1:-NCCT}"   # source, our A
CONTRAST2="${CONTRAST2:-CECT}"   # target, our B

PY="${PYTHON:-python3}"
SLICES="$OUTPUT_PATH/$NAME/images"

log() { printf '\n\033[1;36m[run_syndiff:%s]\033[0m %s\n' "$1" "$2"; }

# SynDiff's dataset.py hardcodes 256: LoadDataSet pads every slice to 256x256
# with pad=(256-shape)/2. At SIZE>256 that pad is negative and np.pad raises; at
# SIZE<256 it silently zero-borders every slice, which shifts the anatomy and
# makes the reassembled volume misalign with the source grid. Neither is
# recoverable downstream, so refuse anything but 256.
require_size_256() {
  [ "$SIZE" = "256" ] || {
    echo "ERROR: SIZE=$SIZE unsupported. SynDiff/dataset.py pads to a hardcoded" >&2
    echo "       256x256; only SIZE=256 round-trips correctly." >&2
    exit 1; }
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

# Newest gen_diffusive_2_<epoch>.pth, or empty if none exist.
latest_ckpt_epoch() {
  ls "$OUTPUT_PATH/$NAME"/gen_diffusive_2_*.pth 2>/dev/null \
    | sed 's#.*gen_diffusive_2_##; s#\.pth$##' \
    | sort -n | tail -1
}

# ---- stages -----------------------------------------------------------------
do_setup() {
  log setup "check SynDiff tree"
  [ -f "$SYNDIFF/train.py" ] || { echo "ERROR: SynDiff/ is missing or empty at $SYNDIFF" >&2; exit 1; }
  echo "  present: $SYNDIFF"

  log setup "preflight the JIT-CUDA toolchain"
  # backbones/up_or_down_sampling.py -> utils/op/{upfirdn2d,fused_act}.py call
  # torch.utils.cpp_extension.load() at import time. Without ninja/nvcc/headers
  # that blows up inside the first epoch; catch it now.
  local miss=0
  command -v ninja >/dev/null 2>&1 || { echo "  MISSING: ninja        (pip install ninja)"; miss=1; }
  command -v nvcc  >/dev/null 2>&1 || { echo "  MISSING: nvcc         (CUDA toolkit; must match torch's CUDA)"; miss=1; }
  "$PY" - <<'EOF' || miss=1
import sys, sysconfig, os
inc = sysconfig.get_paths()["include"]
ok = os.path.exists(os.path.join(inc, "Python.h"))
print(f"  {'ok' if ok else 'MISSING'}: Python.h in {inc}"
      + ("" if ok else f"   (apt install python{sys.version_info.major}.{sys.version_info.minor}-dev)"))
sys.exit(0 if ok else 1)
EOF
  if [ "$miss" = 1 ]; then
    echo
    echo "  ^ SynDiff JIT-compiles CUDA kernels on import; fix the above before training." >&2
    echo "  (verify with: cd $SYNDIFF && $PY -c 'from utils.op import upfirdn2d')" >&2
    exit 1
  fi
  echo "  toolchain ok"

  echo
  echo "  deps: pip install -r $HERE/requirements_syndiff.txt  (in a dedicated env)"
}

do_prep() {
  log prep "our NIfTI -> data_<phase>_{$CONTRAST1,$CONTRAST2}.mat at $DATAROOT"
  require_size_256
  resolve_split
  "$PY" "$HERE/prep_benchmark_data.py" --split "$SPLIT" \
      --format mat --out "$DATAROOT" --size "$SIZE"
}

do_train() {
  for p in train val test; do
    [ -f "$DATAROOT/data_${p}_${CONTRAST1}.mat" ] || {
      echo "ERROR: missing $DATAROOT/data_${p}_${CONTRAST1}.mat (run prep first)" >&2; exit 1; }
  done
  require_size_256
  log train "SynDiff $CONTRAST1 -> $CONTRAST2 ($NUM_EPOCH epochs, ckpt every $SAVE_CKPT_EVERY)"
  cd "$SYNDIFF"
  # --num_process_per_node 1 still goes through NCCL; --local_rank 0 picks the
  # device, so CUDA_VISIBLE_DEVICES is how we select which physical GPU.
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" train.py \
      --image_size "$SIZE" --exp "$NAME" \
      --num_channels "$NUM_CHANNELS" --num_channels_dae "$NUM_CHANNELS_DAE" \
      --ch_mult $CH_MULT --num_timesteps "$NUM_TIMESTEPS" \
      --num_res_blocks "$NUM_RES_BLOCKS" --batch_size "$BATCH" \
      --contrast1 "$CONTRAST1" --contrast2 "$CONTRAST2" \
      --num_epoch "$NUM_EPOCH" --ngf "$NGF" --embedding_type positional \
      --use_ema --ema_decay 0.999 --r1_gamma 1. --z_emb_dim "$Z_EMB_DIM" \
      --lr_d "$LR_D" --lr_g "$LR_G" --lazy_reg 10 \
      --save_content --save_ckpt_every "$SAVE_CKPT_EVERY" \
      --num_process_per_node 1 --local_rank 0 --port_num "$PORT" \
      --input_path "$DATAROOT" --output_path "$OUTPUT_PATH"
}

do_infer() {
  local epoch="$CKPT_EPOCH"
  [ -n "$epoch" ] || epoch="$(latest_ckpt_epoch)"
  if [ -z "$epoch" ]; then
    echo "ERROR: no gen_diffusive_2_*.pth in $OUTPUT_PATH/$NAME" >&2
    echo "  train first; train.py only writes checkpoints every SAVE_CKPT_EVERY=$SAVE_CKPT_EVERY epochs." >&2
    exit 1
  fi
  log infer "gen_diffusive_2 @ epoch $epoch -> *_fake_B.npy"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$HERE/infer_syndiff.py" \
      --syndiff "$SYNDIFF" --input_path "$DATAROOT" \
      --index "$DATAROOT/slice_index.csv" \
      --exp "$NAME" --output_path "$OUTPUT_PATH" --which_epoch "$epoch" \
      --out "$SLICES" --device cuda \
      --contrast1 "$CONTRAST1" --contrast2 "$CONTRAST2" \
      --image_size "$SIZE" --num_channels "$NUM_CHANNELS" \
      --num_channels_dae "$NUM_CHANNELS_DAE" --ch_mult $CH_MULT \
      --num_res_blocks "$NUM_RES_BLOCKS" --num_timesteps "$NUM_TIMESTEPS" \
      --z_emb_dim "$Z_EMB_DIM"
}

do_reassemble() {
  log reassemble "slices -> per-case CECT NIfTI (HU) + manifest.csv"
  # infer_syndiff.py saves the raw generator output, fed in [-1,1] -> neg1_1.
  # Anchored pattern for the same reason as CyTran: our case_ids are dotted
  # DICOM UIDs and a greedy prefix + fixed 4-digit z cannot mis-split them.
  "$PY" "$HERE/reassemble_nifti.py" \
      --index "$DATAROOT/slice_index.csv" \
      --slices_dir "$SLICES" \
      --pattern '^(?P<case>.+)_(?P<z>[0-9]{4})_fake_B\.npy$' \
      --in_range neg1_1 --out "$OUT_NIFTI"
  echo
  echo "  manifest -> $OUT_NIFTI/manifest.csv"
  echo "  score it: cd ../synthetic_CECT && python benchmark.py \\"
  echo "      --weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl \\"
  echo "      --manifest syndiff=$OUT_NIFTI/manifest.csv --baseline ours --out analysis/benchmark"
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

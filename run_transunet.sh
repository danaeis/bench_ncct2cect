#!/usr/bin/env bash
# TransUNet (R50-ViT-B_16) as an NCCT->CECT generator. Thin wrapper over run_backbone.sh.
#   SPLIT=../synthetic_CECT/splits/split.json GPU=0 ./run_transunet.sh all
# Reuses the same ImageNet-21k ViT checkpoint that run_resvit.sh downloads.
exec env ARCH=transunet "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_backbone.sh" "$@"

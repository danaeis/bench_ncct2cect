#!/usr/bin/env bash
# SwinUNETR (MONAI) as an NCCT->CECT generator. Thin wrapper over run_backbone.sh.
#   SPLIT=../synthetic_CECT/splits/split.json GPU=0 ./run_swinunetr.sh all
# See RUN_BACKBONES.md for the recipe and the "trains from scratch" caveat.
exec env ARCH=swinunetr "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_backbone.sh" "$@"

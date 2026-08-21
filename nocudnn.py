#!/usr/bin/env python
"""
Run any training script with cuDNN disabled.

    python nocudnn.py train.py --dataroot ... --name ...

Why a launcher instead of an edit: `torch.backends.cudnn.enabled = False` has to
happen after torch is imported but before the first convolution, and the repos
that need it (CycleGAN, ResViT, CyTran) are vendored upstream code we do not
want to patch for a local environment problem. This wraps them from outside
instead — argv and `__name__ == "__main__"` are preserved, so the target script
cannot tell the difference.

When to use it: only when `preflight_gpu.py` reports
`CUDNN_STATUS_NOT_INITIALIZED` on a card whose SM is >= 7.5 and which has free
memory — i.e. a broken cuDNN runtime rather than an unsupported GPU. Fixing the
environment is better; this keeps you training while you do. Convolutions fall
back to native CUDA kernels, which for these models typically costs somewhere
around 1.5-3x throughput. It cannot rescue an SM < 7.5 card, because there the
failure is in cuDNN's own init and torch refuses the device outright.

`run_*.sh` set this up for you: `DISABLE_CUDNN=1 ./run_cyclegan.sh train`.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)

    target = Path(sys.argv[1])
    if not target.is_file():
        raise SystemExit(f'nocudnn: script not found: {target}')

    import torch
    torch.backends.cudnn.enabled = False
    print(f'[nocudnn] cuDNN disabled (enabled={torch.backends.cudnn.enabled}) '
          f'-> {target}', flush=True)

    # Hand the target script the argv it would have seen if run directly.
    sys.argv = [str(target)] + sys.argv[2:]
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()

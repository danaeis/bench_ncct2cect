#!/usr/bin/env python
"""
Fail in five seconds instead of forty minutes in.

Every runner calls this before training. It checks, in the order the failures
actually happen, the three ways a "working" torch install still cannot train
on the GPU you selected:

  1. `torch.cuda.is_available()` is False.
     A torch built for a NEWER CUDA than the driver supports imports perfectly
     happily and silently reports False. That is how a CycleGAN run spent a day
     printing "Initialized with device cpu".

  2. The GPU's compute capability is not in `torch.cuda.get_arch_list()`.
     Recent wheels drop old architectures. Kernels then fail at launch with
     "no kernel image is available for execution on the device" — after the
     model is built and the data is loaded.

  3. cuDNN cannot initialise or run a convolution on this architecture.
     cuDNN 9.8 removed Maxwell, Pascal and Volta support, so a GTX 1080 Ti
     (sm_61) raises `CUDNN_STATUS_NOT_INITIALIZED` on the FIRST conv — long
     after the checkpoints have loaded and the log looks healthy.

Run it directly to inspect a box:  CUDA_VISIBLE_DEVICES=1 python preflight_gpu.py
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError as e:
        print(f'ERROR: torch is not importable: {e}', file=sys.stderr)
        return 1

    # torch.backends.cudnn.version() is not a passive getter — it calls _init(),
    # which validates every VISIBLE device against the cuDNN build and RAISES.
    # cuDNN 9.x requires SM >= 7.5, so on a Pascal/Volta card this throws before
    # anything else can be reported. Catch it and turn it into a diagnosis.
    try:
        cudnn_v = torch.backends.cudnn.version()
        cudnn_err = None
    except Exception as e:                       # noqa: BLE001 — any init failure
        cudnn_v, cudnn_err = None, e
    if cudnn_err is not None:
        cudnn_desc = f'FAILED TO INIT ({type(cudnn_err).__name__})'
    elif cudnn_v is None:
        cudnn_desc = 'not present in this build'
    else:
        cudnn_desc = str(cudnn_v)
    print(f'  torch {torch.__version__} | built for CUDA {torch.version.cuda} '
          f'| cuDNN {cudnn_desc}', flush=True)

    if not torch.cuda.is_available():
        print('ERROR: CUDA is not available to torch.', file=sys.stderr)
        print('  Compare the driver\'s max CUDA with the build above:', file=sys.stderr)
        print('    nvidia-smi --query-gpu=driver_version --format=csv', file=sys.stderr)
        print('  A CUDA 13.x torch needs driver >= 580; CUDA 12.x needs >= 525.',
              file=sys.stderr)
        print('    pip install torch torchvision --index-url '
              'https://download.pytorch.org/whl/cu126', file=sys.stderr)
        print('  Or set ALLOW_CPU=1 to proceed on CPU anyway (it will not finish).',
              file=sys.stderr)
        return 1

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    sm = f'sm_{cap[0]}{cap[1]}'
    archs = torch.cuda.get_arch_list()
    free, total = torch.cuda.mem_get_info(0)
    print(f'  device: {name} ({sm}) | {free/2**30:.1f} of {total/2**30:.1f} GiB free',
          flush=True)

    if archs and sm not in archs:
        print(f'ERROR: this torch has no kernels for {sm} ({name}).', file=sys.stderr)
        print(f'  compiled architectures: {" ".join(archs)}', file=sys.stderr)
        print('  Pick a GPU whose sm_XX is in that list, or install a torch build '
              'that includes it.', file=sys.stderr)
        return 1

    if cudnn_err is not None:
        print(f'ERROR: cuDNN could not initialise on {name} ({sm}): {cudnn_err}',
              file=sys.stderr)
        if cap < (7, 5):
            print(f'  CAUSE: cuDNN 9.x requires SM >= 7.5 and this card is {sm}. '
                  f'No amount of reinstalling cuDNN will help.', file=sys.stderr)
            print('  Use an SM >= 7.5 GPU, or a much older torch bundling cuDNN 8.x.',
                  file=sys.stderr)
        print('  This card cannot be used by this environment — pick another GPU.',
              file=sys.stderr)
        return 1

    # The real test. Everything above can pass while convolutions still fail.
    import os
    no_cudnn = os.environ.get('DISABLE_CUDNN') == '1'
    if no_cudnn:
        torch.backends.cudnn.enabled = False
        print('  DISABLE_CUDNN=1 — testing the native (non-cuDNN) conv path',
              flush=True)
    try:
        x = torch.randn(2, 1, 64, 64, device='cuda')
        w = torch.nn.Conv2d(1, 8, 3, padding=1).cuda()
        y = w(x)
        torch.cuda.synchronize()
        if tuple(y.shape) != (2, 8, 64, 64):
            print(f'ERROR: conv produced {tuple(y.shape)}, expected (2, 8, 64, 64)',
                  file=sys.stderr)
            return 1
    except Exception as e:
        print(f'ERROR: a convolution failed on {name} ({sm}): '
              f'{type(e).__name__}: {e}', file=sys.stderr)
        if no_cudnn:
            print('  This failed with cuDNN already disabled, so it is not a cuDNN '
                  'problem — suspect the CUDA install or the GPU itself.',
                  file=sys.stderr)
        elif 'NOT_INITIALIZED' in str(e) and cap >= (7, 5):
            # Supported architecture + free memory => the library, not the card.
            print(f'  CAUSE: {sm} is supported by cuDNN {cudnn_v} and '
                  f'{free/2**30:.1f} GiB are free, so this is a BROKEN OR MISMATCHED '
                  f'cuDNN RUNTIME in this environment — torch reports the version it '
                  f'was built against, but loads libcudnn from the filesystem at '
                  f'runtime, and something else is winning.', file=sys.stderr)
            print('  Diagnose which libcudnn is actually being loaded:', file=sys.stderr)
            print('    echo "$LD_LIBRARY_PATH"', file=sys.stderr)
            print('    pip list | grep -i nvidia-cudnn', file=sys.stderr)
            print('    python -c "import torch;torch.randn(1,device=\'cuda\');'
                  'import os;os.system(f\'grep -i cudnn /proc/{os.getpid()}/maps\')"',
                  file=sys.stderr)
            print('  Usual fixes, in order:', file=sys.stderr)
            print('    unset LD_LIBRARY_PATH          # a stale system cuDNN shadowing '
                  'the wheel', file=sys.stderr)
            print('    pip install --force-reinstall nvidia-cudnn-cu12', file=sys.stderr)
            print('    # or build a clean env from requirements_*.txt', file=sys.stderr)
            print('  To keep training NOW at some cost in speed: DISABLE_CUDNN=1',
                  file=sys.stderr)
        else:
            print('  If this is an out-of-memory failure, free the GPU or lower BATCH.',
                  file=sys.stderr)
            print('  Otherwise try DISABLE_CUDNN=1 to fall back to native convs.',
                  file=sys.stderr)
        return 1

    print(f'  {"native" if no_cudnn else "cuDNN"} convolution ok — safe to train')
    return 0


if __name__ == '__main__':
    sys.exit(main())

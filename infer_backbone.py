#!/usr/bin/env python
"""
Run a trained SwinUNETR / TransUNet generator over the test slices.

Writes one `<case>_<zzzz>_fake_B.npy` per slice, holding the raw generator
output in [-1, 1] with **no per-slice renormalisation** — that is what lets
`reassemble_nifti.py --in_range neg1_1` put every slice of a case back on one
shared HU scale. (Saving display PNGs instead would quantise to 8 bits for no
benefit, since nothing downstream looks at them by eye.)

    python infer_backbone.py --arch swinunetr \
        --dataroot ResViT/datasets/vindr --name vindr_swinunetr \
        --out checkpoints/vindr_swinunetr/test_slices
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from backbone_common import ARCHS, ABSliceDataset, build_generator


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arch', choices=ARCHS, required=True)
    ap.add_argument('--dataroot', type=Path, required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--checkpoints_dir', type=Path, default=Path('checkpoints'))
    ap.add_argument('--which_epoch', default='latest',
                    help="'latest' or an epoch number matching <epoch>_net_G.pth")
    ap.add_argument('--out', type=Path, required=True, help='dir for the *_fake_B.npy slices')
    ap.add_argument('--phase', default='test')
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--feature_size', type=int, default=48)
    ap.add_argument('--n_slices', type=int, default=1,
                    help='must match the value used at training time')
    ap.add_argument('--transunet_dir', type=Path, default=Path('TransUNet'))
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu'
                          else 'cpu')
    ckpt = args.checkpoints_dir / args.name / f'{args.which_epoch}_net_G.pth'
    if not ckpt.is_file():
        raise SystemExit(f'checkpoint not found: {ckpt}')

    # The architecture must be rebuilt exactly as trained. TransUNet's ImageNet
    # init is irrelevant here — load_state_dict overwrites all of it — so skip
    # the 400 MB .npz read and build the bare network.
    netG = build_generator(args.arch, args.size, args.n_slices, 1, args.feature_size,
                           args.transunet_dir, None).to(device)
    netG.load_state_dict(torch.load(ckpt, map_location=device))
    netG.eval()
    print(f'[model] {args.arch} <- {ckpt}')

    ds = ABSliceDataset(args.dataroot, args.phase, args.size, args.n_slices)
    ld = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f'[data] {len(ds)} {args.phase} slices -> {args.out}')

    written = 0
    with torch.no_grad():
        for A, _, stems in ld:
            fake = netG(A.to(device)).cpu().numpy()          # (N,1,H,W) in [-1,1]
            for k, stem in enumerate(stems):
                np.save(args.out / f'{stem}_fake_B.npy', fake[k, 0].astype(np.float32))
                written += 1

    print(f'[done] {written} slices -> {args.out}')
    print('  reassemble:  python reassemble_nifti.py --index '
          f'{args.dataroot}/slice_index.csv --slices_dir {args.out} \\')
    print("      --pattern '^(?P<case>.+)_(?P<z>[0-9]{4})_fake_B\\.npy$' "
          '--in_range neg1_1 --out <nifti_dir>')


if __name__ == '__main__':
    main()

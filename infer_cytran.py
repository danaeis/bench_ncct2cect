#!/usr/bin/env python
"""
Run a trained CyTran generator over the TEST slices and write per-slice outputs
that reassemble_nifti.py can stitch back into per-case CECT NIfTI.

Why this exists: CyTran's own `test.py` is not a general inference script — it
hardcodes the Coltea-Lung-CT-100W DICOM tree, recomputes MAE/RMSE/SSIM in the
ARTERIAL->NATIVE direction, and never writes an image. So we load the trained
G_A weights into a bare ConvTransformer and do the forward pass ourselves.

Direction: our prep makes A = NCCT and B = CECT, and training runs `--direction
AtoB`, so G_A is the NCCT->CECT generator — that is the one loaded here
(`<epoch>_net_G_A.pth`).

The generator's last layer is a plain conv (no tanh), so outputs are not bounded
to [-1,1]; we save them raw as .npy and let reassemble_nifti.py clip with
`--in_range neg1_1`, matching the [-1,1] range the dataset feeds in.

Usage:
    python infer_cytran.py \
        --cytran CyTran --index CyTran/datasets/vindr/slice_index.csv \
        --checkpoint CyTran/checkpoints/vindr_cytran/latest_net_G_A.pth \
        --out CyTran/results/vindr_cytran/images
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cytran', type=Path, required=True, help='CyTran repo root')
    ap.add_argument('--index', type=Path, required=True, help='slice_index.csv from prep')
    ap.add_argument('--checkpoint', type=Path, required=True, help='*_net_G_A.pth')
    ap.add_argument('--out', type=Path, required=True, help='dir for *_fake_B.npy')
    ap.add_argument('--phase', default='test')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    # must match the training flags (see run_cytran.sh)
    ap.add_argument('--input_nc', type=int, default=1)
    ap.add_argument('--ngf_cytran', type=int, default=16)
    ap.add_argument('--n_downsampling', type=int, default=3)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--heads', type=int, default=6)
    args = ap.parse_args()

    sys.path.insert(0, str(args.cytran.resolve()))
    from models.conv_transformer import ConvTransformer   # noqa: E402

    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available())
                          else 'cpu')
    if str(device) != args.device:
        print(f'[warn] CUDA unavailable — falling back to {device}')

    net = ConvTransformer(input_nc=args.input_nc, n_downsampling=args.n_downsampling,
                          depth=args.depth, heads=args.heads, dropout=0.0,
                          ngf=args.ngf_cytran).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if hasattr(state, '_metadata'):
        del state._metadata
    net.load_state_dict(state)
    net.eval()
    print(f'[loaded] {args.checkpoint}')

    rows = [r for r in csv.DictReader(args.index.open()) if r['phase'] == args.phase]
    if not rows:
        raise SystemExit(f'no {args.phase} rows in {args.index}')
    args.out.mkdir(parents=True, exist_ok=True)

    written = 0
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start:start + args.batch_size]
        batch, names = [], []
        for r in chunk:
            # out_ref is the absolute path recorded at prep time; fall back to
            # the index-relative location if the dataset tree was moved since.
            p = Path(r['out_ref'])
            if not p.exists():
                p = args.index.parent / r['phase'] / f"{r['case_id']}_{int(r['z']):04d}.png"
            im = np.asarray(Image.open(p).convert('L'), dtype=np.float32)
            w = im.shape[1] // 2
            batch.append(im[:, :w] / 127.5 - 1.0)          # A = NCCT, [-1,1]
            names.append(f"{r['case_id']}_{int(r['z']):04d}_fake_B.npy")
        x = torch.from_numpy(np.expand_dims(np.stack(batch), 1)).float().to(device)
        with torch.no_grad():
            y = net(x).cpu().numpy()[:, 0]                 # (B,H,W), ~[-1,1]
        for name, sl in zip(names, y):
            np.save(args.out / name, sl.astype(np.float32))
            written += 1
        if start % (args.batch_size * 50) == 0:
            print(f'  {written}/{len(rows)} slices')

    print(f'[written] {written} slices → {args.out}')
    print('Stitch with: reassemble_nifti.py --slice_suffix _fake_B --in_range neg1_1')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
Convert our VinDr NCCT/CECT NIfTI volumes into the input formats the benchmark
repos expect — so every model trains on the SAME cases, our own data.

Reads the shared split (`../synthetic_CECT/splits/split.json`) and emits, per
phase (train/val/test), 2-D axial slices in one of two layouts:

  --format pix2pix   → <out>/<phase>/<case>_<z>.png, side-by-side [A|B] where
                       A = NCCT, B = CECT. This is the classic pix2pix aligned
                       format that ResViT and CyTran both read (data/aligned_dataset.py).
  --format mat       → <out>/data_<phase>_NCCT.mat and data_<phase>_CECT.mat,
                       HDF5 with variable `data_fs`, shape (N, size, size) in
                       [0,1] — the layout SynDiff's LoadDataSet expects.

Always writes <out>/slice_index.csv: one row per emitted slice mapping it back to
its case, z-index, native shape and the real-CECT/seg paths. `reassemble_nifti.py`
uses it to turn a model's per-slice outputs back into a full NIfTI + a scoring
manifest.

All slices are windowed to the shared HU range → [0,1] and resized to --size
(default 256) so both repos get a uniform input; the native shape is recorded so
reassembly restores the original grid for fair scoring.

Every axial slice is emitted (no tissue filtering) so the reconstructed TEST
volume is complete; pass --min_tissue_frac >0 to drop near-empty slices from the
TRAIN/VAL sets only (a training-efficiency choice that never touches test).

Usage:
    python prep_benchmark_data.py --split ../synthetic_CECT/splits/split.json \
        --format pix2pix --out ResViT/datasets/vindr --size 256
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import nibabel as nib
from PIL import Image


def _load(p: str) -> np.ndarray:
    return np.asanyarray(nib.load(p).dataobj).astype(np.float32)


def _to_unit(vol: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    v = np.clip(vol, hu_min, hu_max)
    return (v - hu_min) / (hu_max - hu_min)


def _resize01(sl: np.ndarray, size: int) -> np.ndarray:
    """[0,1] slice → size×size [0,1] via PIL bicubic (uint8 round-trip is fine at
    8-bit display precision, which is all the PNG path carries anyway)."""
    im = Image.fromarray((np.clip(sl, 0, 1) * 255).astype(np.uint8))
    im = im.resize((size, size), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', type=Path, required=True)
    ap.add_argument('--format', choices=['pix2pix', 'mat'], required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--slice_axis', type=int, default=2, help='axial axis (default 2)')
    ap.add_argument('--min_tissue_frac', type=float, default=0.0,
                    help='drop train/val slices with < this fraction of >0.1 voxels')
    ap.add_argument('--phases', nargs='+', default=['train', 'val', 'test'])
    args = ap.parse_args()

    split = json.loads(args.split.read_text())
    hu_min, hu_max = split.get('hu_min', -200.0), split.get('hu_max', 400.0)
    target_phase = split.get('target_phase', 'venous')
    args.out.mkdir(parents=True, exist_ok=True)

    index_rows: List[dict] = []
    mat_buffers = {}    # (phase, 'NCCT'|'CECT') -> list of slices

    for phase in args.phases:
        cases = split.get(phase, [])
        if args.format == 'pix2pix':
            (args.out / phase).mkdir(parents=True, exist_ok=True)
        print(f'[{phase}] {len(cases)} cases')
        keep_all = phase == 'test'          # test volumes must reconstruct fully

        for case in cases:
            try:
                ncct = _to_unit(_load(case['ncct']), hu_min, hu_max)
                cect = _to_unit(_load(case['cect']), hu_min, hu_max)
            except Exception as e:
                print(f"  skip {case['case_id']}: {e}")
                continue
            if ncct.shape != cect.shape:
                print(f"  skip {case['case_id']}: shape {ncct.shape} vs {cect.shape}")
                continue

            A_all = np.moveaxis(ncct, args.slice_axis, 0)
            B_all = np.moveaxis(cect, args.slice_axis, 0)
            H, W = A_all.shape[1], A_all.shape[2]
            nz = A_all.shape[0]

            for z in range(nz):
                b = B_all[z]
                if (not keep_all) and args.min_tissue_frac > 0:
                    if float((b > 0.1).mean()) < args.min_tissue_frac:
                        continue
                a_r = _resize01(A_all[z], args.size)
                b_r = _resize01(b, args.size)
                cid = case['case_id']

                if args.format == 'pix2pix':
                    ab = np.concatenate([a_r, b_r], axis=1)      # [A | B]
                    fn = args.out / phase / f'{cid}_{z:04d}.png'
                    Image.fromarray((ab * 255).astype(np.uint8), mode='L').save(fn)
                    out_ref = str(fn)
                else:
                    mat_buffers.setdefault((phase, 'NCCT'), []).append(a_r)
                    mat_buffers.setdefault((phase, 'CECT'), []).append(b_r)
                    out_ref = f'{phase}#{len(mat_buffers[(phase,"NCCT")])-1}'  # row index

                index_rows.append({
                    'phase': phase, 'case_id': cid, 'z': z,
                    'native_h': H, 'native_w': W, 'native_z': nz,
                    'real_path': case['cect'], 'mask_path': case.get('seg') or '',
                    'target_phase': target_phase, 'out_ref': out_ref,
                })

    if args.format == 'mat':
        import h5py            # lazy: only the .mat path needs it
        for (phase, contrast), slices in mat_buffers.items():
            arr = np.stack(slices).astype(np.float32)            # (N, size, size), [0,1]
            fn = args.out / f'data_{phase}_{contrast}.mat'
            with h5py.File(fn, 'w') as f:
                f.create_dataset('data_fs', data=arr)
            print(f'  wrote {fn}  {arr.shape}')

    idx = args.out / 'slice_index.csv'
    with idx.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0]))
        w.writeheader(); w.writerows(index_rows)
    print(f'[written] {len(index_rows)} slices, index → {idx}')
    print(f'NCCT=A (input), CECT=B (target). Train {args.format} '
          f"({'AtoB' if args.format=='pix2pix' else 'NCCT→CECT'}).")


if __name__ == '__main__':
    main()

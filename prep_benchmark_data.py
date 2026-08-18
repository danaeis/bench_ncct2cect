#!/usr/bin/env python
"""
Convert our VinDr NCCT/CECT NIfTI volumes into the input formats the benchmark
repos expect — so every model trains on the SAME cases, our own data.

Reads the shared split (`../synthetic_CECT/splits/split.json`) and emits, per
phase (train/val/test), 2-D axial slices in one of two layouts:

  --format pix2pix   → <out>/<phase>/<case>_<z>.png, side-by-side [A|B] where
                       A = NCCT, B = CECT. This is the classic pix2pix aligned
                       format that ResViT and CyTran both read (data/aligned_dataset.py).
                       SwinUNETR and TransUNet read this same layout through
                       backbone_common.py, so they need no format of their own.
  --format cyclegan  → <out>/<phase>A/<case>_<z>.png (NCCT) and
                       <out>/<phase>B/<case>_<z>.png (CECT), i.e. the two domains
                       in separate directories. This is what CycleGAN's
                       data/unaligned_dataset.py expects (dir_A = phase+"A").
                       The slices are still perfectly paired on disk — CycleGAN
                       simply ignores the pairing and samples B at random, which
                       is exactly the unpaired baseline we want to measure.
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


def _abspath(raw: str, split: Path) -> str:
    """A source path from split.json -> an absolute path string.

    split.json stores paths relative to the directory the split was written from
    (ours look like `../sample_data_reg/...`). Recording them verbatim in
    slice_index.csv makes every later consumer — reassemble_nifti.py, and
    benchmark.py through the manifest it writes — depend on being run from that
    same CWD, which they are not. Resolve once, here.
    """
    if not raw:
        return ''
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    for root in (Path.cwd(), split.resolve().parent, split.resolve().parent.parent):
        if (root / p).exists():
            return str((root / p).resolve())
    return str(p.resolve())          # keep going; the loader will report it


def _to_unit(vol: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    v = np.clip(vol, hu_min, hu_max)
    return (v - hu_min) / (hu_max - hu_min)


class _MatWriter:
    """Streaming writer for SynDiff's `data_<phase>_<contrast>.mat`.

    Buffers a fixed number of slices and extends a resizable HDF5 dataset, so
    peak RAM is one chunk rather than the whole split.

    WHY THIS IS NOT A LIST. The obvious version — append every slice to a list,
    `np.stack()` at the end — held all three phases × both contrasts in memory
    at once and then allocated a second full copy during the stack. On our VinDr
    split (25,251 train + ~5,200 val + ~5,200 test slices at 256², float32) that
    is 18.7 GB resident peaking at 25.3 GB, which is enough to OOM a shared box
    or drive it into swap-thrash. Streaming keeps it flat at ~64 MB.
    """

    CHUNK = 256          # slices per HDF5 extend; 256 * 256² * 4B = 64 MB

    def __init__(self, path: Path, size: int):
        import h5py                     # lazy: only the .mat path needs it
        self._f = h5py.File(path, 'w')
        self._ds = self._f.create_dataset(
            'data_fs', shape=(0, size, size), maxshape=(None, size, size),
            dtype='float32', chunks=(1, size, size))
        self._buf: List[np.ndarray] = []
        self.n = 0

    def append(self, sl: np.ndarray) -> int:
        """Queue one slice; returns its row index in the finished dataset."""
        self._buf.append(sl)
        self.n += 1
        if len(self._buf) >= self.CHUNK:
            self.flush()
        return self.n - 1

    def flush(self) -> None:
        if not self._buf:
            return
        arr = np.stack(self._buf).astype(np.float32)
        k = self._ds.shape[0]
        self._ds.resize(k + arr.shape[0], axis=0)
        self._ds[k:k + arr.shape[0]] = arr
        self._buf = []

    def close(self) -> None:
        self.flush()
        self._f.close()


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
    ap.add_argument('--format', choices=['pix2pix', 'cyclegan', 'mat'], required=True)
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
    mat_writers = {}    # (phase, 'NCCT'|'CECT') -> _MatWriter, opened on first use

    for phase in args.phases:
        cases = split.get(phase, [])
        if args.format == 'pix2pix':
            (args.out / phase).mkdir(parents=True, exist_ok=True)
        elif args.format == 'cyclegan':
            (args.out / f'{phase}A').mkdir(parents=True, exist_ok=True)
            (args.out / f'{phase}B').mkdir(parents=True, exist_ok=True)
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
                elif args.format == 'cyclegan':
                    # Same slices, two dirs. out_ref points at the A-side file,
                    # because CycleGAN names its generated slices after the A
                    # input it was fed.
                    fa = args.out / f'{phase}A' / f'{cid}_{z:04d}.png'
                    fb = args.out / f'{phase}B' / f'{cid}_{z:04d}.png'
                    Image.fromarray((a_r * 255).astype(np.uint8), mode='L').save(fa)
                    Image.fromarray((b_r * 255).astype(np.uint8), mode='L').save(fb)
                    out_ref = str(fa)
                else:
                    for contrast, sl in (('NCCT', a_r), ('CECT', b_r)):
                        key = (phase, contrast)
                        if key not in mat_writers:
                            mat_writers[key] = _MatWriter(
                                args.out / f'data_{phase}_{contrast}.mat', args.size)
                        row = mat_writers[key].append(sl)
                    out_ref = f'{phase}#{row}'          # row index, same for both

                index_rows.append({
                    'phase': phase, 'case_id': cid, 'z': z,
                    'native_h': H, 'native_w': W, 'native_z': nz,
                    'real_path': _abspath(case['cect'], args.split),
                    'mask_path': _abspath(case.get('seg') or '', args.split),
                    'target_phase': target_phase, 'out_ref': out_ref,
                })

    for (phase, contrast), w in mat_writers.items():
        n = w.n
        w.close()
        print(f'  wrote {args.out}/data_{phase}_{contrast}.mat  ({n}, {args.size}, {args.size})')

    idx = args.out / 'slice_index.csv'
    with idx.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0]))
        w.writeheader(); w.writerows(index_rows)
    print(f'[written] {len(index_rows)} slices, index → {idx}')
    print(f'NCCT=A (input), CECT=B (target). Train {args.format} '
          f"({'AtoB' if args.format in ('pix2pix', 'cyclegan') else 'NCCT→CECT'}).")


if __name__ == '__main__':
    main()

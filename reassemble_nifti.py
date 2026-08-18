#!/usr/bin/env python
"""
Turn a model's per-slice outputs back into per-case CECT NIfTI + a scoring
manifest, so `../synthetic_CECT/benchmark.py` can score it.

Inverse of prep_benchmark_data.py. Reads that tool's slice_index.csv (which knows
each slice's case, z-index and native grid) plus a directory of the model's output
slices, then per case: resizes each slice back to the native grid, stacks them in
z, de-normalises [0,1]→HU, and saves a NIfTI on the real CECT's affine/grid. Writes
manifest.csv with the columns benchmark.py expects.

Output-slice filename convention: the model must name each output so its
`<case>_<z>` is recoverable. By default we match `*<case>_<zzzz>*` from the index's
out_ref basename; override with --pattern if a repo renames files.

Only the TEST phase is reassembled (that is what gets scored).

Usage:
    python reassemble_nifti.py \
        --index ResViT/datasets/vindr/slice_index.csv \
        --slices_dir ResViT/results/vindr/test_latest/images \
        --slice_suffix _fake_B --out ResViT/results/vindr_nifti \
        --in_range 0_255
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import nibabel as nib
from PIL import Image


def _read_slice(path: Path, in_range: str) -> np.ndarray:
    """Load a model-output slice → [0,1] float. PNGs are display-range [0,255];
    some repos save [-1,1] npy."""
    if path.suffix == '.npy':
        a = np.load(path).astype(np.float32).squeeze()
    else:
        a = np.asarray(Image.open(path).convert('L'), dtype=np.float32)
    if in_range == '0_255':
        return np.clip(a / 255.0, 0, 1)
    if in_range == 'neg1_1':
        return np.clip((a + 1) / 2, 0, 1)
    return np.clip(a, 0, 1)          # already 0_1


def _resize(sl: np.ndarray, h: int, w: int) -> np.ndarray:
    im = Image.fromarray((np.clip(sl, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def _candidate_roots(index: Path, extra: Optional[Path]) -> List[Path]:
    """Directories a RELATIVE path in slice_index.csv might be relative to.

    The index stores whatever `split.json` held, and our splits hold paths like
    `../sample_data_reg/...` — relative to the repo root, i.e. to the CWD that
    ran `prep`. Reassembly does not necessarily run from there (a runner that
    `cd`s into a vendored repo for training leaves the CWD there), so resolve
    against every plausible root instead of trusting the CWD.
    """
    roots = [extra] if extra else []
    roots += [Path.cwd(), Path(__file__).resolve().parent]   # repo root
    d = index.resolve().parent
    for _ in range(4):                       # e.g. CyTran/datasets/vindr -> ...
        roots.append(d)
        if d == d.parent:
            break
        d = d.parent
    out, seen = [], set()
    for r in roots:
        r = Path(r).resolve()
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _resolve_data_path(raw: str, roots: List[Path]) -> Path:
    """A path from the index -> an existing absolute path, or a clear error."""
    p = Path(raw)
    if p.is_file():
        return p.resolve()
    if not p.is_absolute():
        for r in roots:
            if (r / p).is_file():
                return (r / p).resolve()
    tried = '\n    '.join(str(r / p) for r in roots) if not p.is_absolute() else str(p)
    raise SystemExit(
        f"ERROR: source volume from slice_index.csv not found: {raw}\n"
        f"  tried:\n    {tried}\n"
        "  The index stores the paths that were in split.json. If they are "
        "relative, run this from the directory prep was run in, or pass "
        "--data_root <dir they are relative to>.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--index', type=Path, required=True, help='slice_index.csv from prep')
    ap.add_argument('--slices_dir', type=Path, required=True, help="model's output slices")
    ap.add_argument('--out', type=Path, required=True, help='dir for reassembled NIfTI + manifest')
    ap.add_argument('--slice_suffix', default='',
                    help='e.g. _fake_B — the repo\'s generated-slice suffix')
    ap.add_argument('--pattern', default=None,
                    help='custom regex with (?P<case>) and (?P<z>) groups; overrides default matching')
    ap.add_argument('--in_range', choices=['0_255', 'neg1_1', '0_1'], default='0_255')
    ap.add_argument('--slice_axis', type=int, default=2)
    ap.add_argument('--hu_min', type=float, default=-200.0)
    ap.add_argument('--hu_max', type=float, default=400.0)
    ap.add_argument('--phase', default='test')
    ap.add_argument('--data_root', type=Path, default=None,
                    help='directory that RELATIVE real/mask paths in the index are '
                         'relative to (default: try CWD, the repo root and the '
                         "index's parents)")
    args = ap.parse_args()

    roots = _candidate_roots(args.index, args.data_root)

    rows = [r for r in csv.DictReader(args.index.open()) if r['phase'] == args.phase]
    by_case = defaultdict(list)
    for r in rows:
        by_case[r['case_id']].append(r)

    # Locate each output slice file. Index every candidate once, then match.
    files = [p for p in args.slices_dir.rglob('*')
             if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.npy')]
    if args.pattern:
        rx = re.compile(args.pattern)
        lut = {}
        for p in files:
            m = rx.search(p.name)
            if m:
                lut[(m.group('case'), int(m.group('z')))] = p
    else:
        # default: filename contains "<case>_<zzzz>" and the required suffix
        lut = {}
        for p in files:
            if args.slice_suffix and args.slice_suffix not in p.name:
                continue
            m = re.search(r'(?P<case>.+?)_(?P<z>\d{3,4})(?=[_.])', p.name)
            if m:
                lut[(m.group('case'), int(m.group('z')))] = p

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for cid, cases in by_case.items():
        ref = cases[0]
        H, W, Z = int(ref['native_h']), int(ref['native_w']), int(ref['native_z'])
        vol01 = np.zeros((Z, H, W), dtype=np.float32)      # z-major, filled per slice
        filled = 0
        for r in cases:
            z = int(r['z'])
            p = lut.get((cid, z))
            if p is None:
                continue
            vol01[z] = _resize(_read_slice(p, args.in_range), H, W)
            filled += 1
        if filled == 0:
            print(f"  {cid}: no output slices matched — skipped")
            continue
        if filled < Z:
            print(f"  {cid}: only {filled}/{Z} slices matched (missing → window floor)")

        vol01 = np.moveaxis(vol01, 0, args.slice_axis)     # back to native orientation
        vol_hu = vol01 * (args.hu_max - args.hu_min) + args.hu_min

        # Save on the real CECT's affine/header so gen and real share a grid.
        real_path = _resolve_data_path(ref['real_path'], roots)
        real_img = nib.load(real_path)
        gen_path = (args.out / f'{cid}_syn.nii.gz').resolve()
        nib.save(nib.Nifti1Image(vol_hu.astype(np.float32), real_img.affine, real_img.header),
                 gen_path)
        # Absolute paths in the manifest: benchmark.py runs from ../synthetic_CECT,
        # where a relative path from our split would point somewhere else entirely.
        mask_path = (str(_resolve_data_path(ref['mask_path'], roots))
                     if ref['mask_path'] else '')
        manifest.append({'gen_path': str(gen_path), 'real_path': str(real_path),
                         'mask_path': mask_path, 'target_phase': ref['target_phase']})

    mf = args.out / 'manifest.csv'
    with mf.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['gen_path', 'real_path', 'mask_path', 'target_phase'])
        w.writeheader(); w.writerows(manifest)
    print(f'[written] {len(manifest)} volumes + manifest → {mf}')
    print('Score with: cd ../synthetic_CECT && python benchmark.py '
          f'--weights orgFeatXGB_CTPhase/xgb_vindr_full.pkl --manifest <name>={mf} ...')


if __name__ == '__main__':
    main()

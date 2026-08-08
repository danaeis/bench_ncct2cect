# This file is COPIED into CyTran/data/vindr_dataset.py by run_cytran.sh (setup).
# It lives here, in the parent repo, because the CyTran submodule is tracked by
# SHA — anything written inside it cannot be committed.
#
# CyTran's stock loader (data/ct_dataset.py) unpickles a dict of DICOM paths from
# the Coltea-Lung-CT-100W release and rescales raw stored values with
# `x/1e3 - 1`. Our data is VinDr NIfTI, prepped by prep_benchmark_data.py into
# pix2pix-style side-by-side [A|B] PNGs (A = NCCT, B = CECT), already windowed to
# the shared HU range and resized. This dataset reads those directly.
#
# Enabled with `--dataset_mode vindr` (data/__init__.py maps the name to
# vindr_dataset.VindrDataset).

import os
from glob import glob

import numpy as np
from PIL import Image

from data.base_dataset import BaseDataset


class VindrDataset(BaseDataset):
    """Paired NCCT->CECT slices from prep_benchmark_data.py --format pix2pix.

    Each file <dataroot>/<phase>/<case>_<zzzz>.png is a grayscale image of width
    2*S holding [A | B]: A = NCCT (input), B = CECT (target). Pixels are the
    shared HU window mapped to [0,255]; we hand the model [-1,1], which is the
    range every other pix2pix/CycleGAN-family loader in this benchmark uses and
    which infer_cytran.py inverts on the way out.
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        phase = getattr(opt, 'phase', 'train')
        self.dir = os.path.join(opt.dataroot, phase)
        self.paths = sorted(glob(os.path.join(self.dir, '*.png')))
        if not self.paths:
            raise FileNotFoundError(
                f'no slices in {self.dir} — run `run_cytran.sh prep` first '
                '(prep_benchmark_data.py --format pix2pix)')
        print(f'[vindr] {len(self.paths)} {phase} slices from {self.dir}')

    def __getitem__(self, index):
        path = self.paths[index]
        im = np.asarray(Image.open(path).convert('L'), dtype=np.float32)
        h, w2 = im.shape
        w = w2 // 2
        A = im[:, :w] / 127.5 - 1.0        # NCCT, [-1,1]
        B = im[:, w:] / 127.5 - 1.0        # CECT, [-1,1]
        return {
            'A': np.expand_dims(A, 0),     # (1,H,W)
            'B': np.expand_dims(B, 0),
            'A_paths': path,
            'B_paths': path,
        }

    def __len__(self):
        return len(self.paths)

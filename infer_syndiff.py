#!/usr/bin/env python
"""
Run a trained SynDiff diffusive generator over the TEST slices and write
per-slice outputs that reassemble_nifti.py can stitch back into per-case CECT
NIfTI.

Why this exists — SynDiff's own `test.py` cannot be used for our benchmark:

  * it hardcodes `transforms.CenterCrop((256, 152))`, the IXI/BRATS field of
    view, then assigns the cropped (256,152) result into a preallocated
    (256,256,N) buffer. That is a straight ValueError on any dataset, ours
    included, and it would crop away a third of the CT anyway.
  * it renormalises every slice by its own max (`fake/fake.max()`). Per-slice
    rescaling destroys the global intensity scale, so slices can no longer be
    mapped back to a common HU window — fatal for volume reconstruction.
  * it writes .jpg triptychs and one big im_syn.mat, neither of which carries
    the case_id/z needed to rebuild a volume.

Direction. Our prep writes contrast1 = NCCT (source) and contrast2 = CECT
(target). In SynDiff's second test loop the roles are `source_data = x`
(contrast1) and the network is `gen_diffusive_2`, i.e. NCCT -> CECT. That is our
direction, so `gen_diffusive_2_<epoch>.pth` is the checkpoint loaded here.
`gen_diffusive_1` is the reverse (CECT -> NCCT) and is ignored.

Transpose. SynDiff's LoadDataSet applies `np.transpose(data, (0,2,1))` to any
3-D .mat — a MATLAB v7.3 column-major convention. Our .mat is written by h5py
and needs no such swap, but the loader applies it unconditionally, so the model
sees every slice transposed. Both contrasts get the same treatment so training
is self-consistent; we transpose back on the way out so the saved slices match
the orientation reassemble_nifti.py expects.

Range. Output is left in the raw [-1,1] the model produces (no per-slice max
rescaling) and saved as .npy, so reassembly uses `--in_range neg1_1`.

Usage:
    python infer_syndiff.py \
        --syndiff SynDiff --input_path SynDiff/datasets/vindr \
        --index SynDiff/datasets/vindr/slice_index.csv \
        --exp vindr_syndiff --output_path SynDiff/results \
        --which_epoch 50 --out SynDiff/results/vindr_syndiff/images
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---- our wiring ----------------------------------------------------------
    ap.add_argument('--syndiff', type=Path, required=True, help='SynDiff repo root')
    ap.add_argument('--input_path', type=Path, required=True,
                    help='dir holding data_test_<contrast>.mat')
    ap.add_argument('--index', type=Path, required=True, help='slice_index.csv from prep')
    ap.add_argument('--out', type=Path, required=True, help='dir for *_fake_B.npy')
    ap.add_argument('--output_path', type=Path, required=True,
                    help='training output root (holds <exp>/gen_diffusive_2_<epoch>.pth)')
    ap.add_argument('--exp', required=True, help='experiment name used at training')
    ap.add_argument('--which_epoch', type=int, required=True)
    ap.add_argument('--phase', default='test')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--contrast1', default='NCCT', help='source contrast (our A)')
    ap.add_argument('--contrast2', default='CECT', help='target contrast (our B)')

    # ---- architecture: MUST match the training flags in run_syndiff.sh -------
    # Defaults mirror SynDiff test.py so an unset flag never silently changes
    # the network shape relative to upstream.
    ap.add_argument('--image_size', type=int, default=256)
    ap.add_argument('--num_channels', type=int, default=2, help='in-channels: noise + source')
    ap.add_argument('--num_channels_dae', type=int, default=64)
    ap.add_argument('--ch_mult', nargs='+', type=int, default=[1, 1, 2, 2, 4, 4])
    ap.add_argument('--num_res_blocks', type=int, default=2)
    ap.add_argument('--num_timesteps', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--nz', type=int, default=100)
    ap.add_argument('--z_emb_dim', type=int, default=256)
    ap.add_argument('--t_emb_dim', type=int, default=256)
    ap.add_argument('--n_mlp', type=int, default=3)
    ap.add_argument('--embedding_type', default='positional', choices=['positional', 'fourier'])
    ap.add_argument('--fourier_scale', type=float, default=16.)
    ap.add_argument('--attn_resolutions', default=(16,))
    ap.add_argument('--dropout', type=float, default=0.)
    ap.add_argument('--resblock_type', default='biggan')
    ap.add_argument('--progressive', default='none', choices=['none', 'output_skip', 'residual'])
    ap.add_argument('--progressive_input', default='residual',
                    choices=['none', 'input_skip', 'residual'])
    ap.add_argument('--progressive_combine', default='sum', choices=['sum', 'cat'])
    ap.add_argument('--fir_kernel', default=[1, 3, 3, 1])
    ap.add_argument('--beta_min', type=float, default=0.1)
    ap.add_argument('--beta_max', type=float, default=20.)
    ap.add_argument('--use_geometric', action='store_true', default=False)
    ap.add_argument('--not_use_tanh', action='store_true', default=False)
    # store_false flags: present on the command line means DISABLE, matching upstream
    ap.add_argument('--centered', action='store_false', default=True)
    ap.add_argument('--resamp_with_conv', action='store_false', default=True)
    ap.add_argument('--conditional', action='store_false', default=True)
    ap.add_argument('--fir', action='store_false', default=True)
    ap.add_argument('--skip_rescale', action='store_false', default=True)
    return ap


def load_index(index_csv: Path, phase: str):
    """Row index within the phase -> output basename, in prep's emission order.

    prep_benchmark_data.py appends to the .mat buffer and to slice_index.csv in
    lockstep and records out_ref='<phase>#<row>', so the row number is the
    authoritative link between a .mat row and its (case_id, z).
    """
    by_row = {}
    with index_csv.open() as f:
        for r in csv.DictReader(f):
            if r['phase'] != phase:
                continue
            ref = r['out_ref']
            if '#' not in ref:
                raise SystemExit(
                    f'--index {index_csv} was written by `--format pix2pix`; SynDiff '
                    f'needs the index from `--format mat` (out_ref like "test#0").')
            row = int(ref.split('#', 1)[1])
            if row in by_row:
                raise SystemExit(f'duplicate row {row} in {index_csv}')
            by_row[row] = f"{r['case_id']}_{int(r['z']):04d}_fake_B.npy"
    if not by_row:
        raise SystemExit(f'no rows for phase={phase} in {index_csv}')
    missing = set(range(len(by_row))) - set(by_row)
    if missing:
        raise SystemExit(f'{index_csv} row numbers are not contiguous; missing {sorted(missing)[:5]}')
    return [by_row[i] for i in range(len(by_row))]


def main():
    args = build_parser().parse_args()

    root = args.syndiff.resolve()
    sys.path.insert(0, str(root))
    # Reuse upstream's diffusion math and checkpoint loader verbatim so our
    # sampling cannot drift from theirs. test.py guards its CLI behind
    # __main__, so importing it is side-effect free.
    try:
        from backbones.ncsnpp_generator_adagn import NCSNpp          # noqa: E402
        from dataset import CreateDatasetSynthesis                    # noqa: E402
        from test import (Posterior_Coefficients, get_time_schedule,  # noqa: E402
                          load_checkpoint, sample_from_model)
    except (OSError, ImportError, RuntimeError) as e:
        # utils/op JIT-builds CUDA kernels at import; without nvcc/ninja/Python.h
        # this surfaces as a confusing CUDA_HOME or ninja error far from the cause.
        raise SystemExit(
            f'failed to import SynDiff backbones: {e}\n'
            f'  SynDiff JIT-compiles CUDA kernels on import and needs nvcc + ninja\n'
            f'  + pythonX.Y-dev, on a machine with a GPU. Run `./run_syndiff.sh setup`\n'
            f'  to check the toolchain.')

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('[warn] CUDA unavailable — falling back to cpu')
        args.device = 'cpu'
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    names = load_index(args.index, args.phase)

    dataset = CreateDatasetSynthesis(args.phase, str(args.input_path),
                                     args.contrast1, args.contrast2)
    if len(dataset) != len(names):
        raise SystemExit(
            f'{len(dataset)} slices in data_{args.phase}_*.mat but {len(names)} '
            f'{args.phase} rows in {args.index} — regenerate both with the same '
            f'`prep_benchmark_data.py --format mat` run.')
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)

    gen = NCSNpp(args).to(device)
    ckpt_tmpl = str(args.output_path.resolve() / args.exp / '{}_{}.pth')
    ckpt = ckpt_tmpl.format('gen_diffusive_2', args.which_epoch)
    if not Path(ckpt).exists():
        avail = sorted(p.name for p in (args.output_path / args.exp).glob('gen_diffusive_2_*.pth')) \
            if (args.output_path / args.exp).is_dir() else []
        raise SystemExit(f'checkpoint not found: {ckpt}\n'
                         f'  available: {avail or "(none — train first)"}')
    load_checkpoint(ckpt_tmpl, gen, 'gen_diffusive_2', epoch=str(args.which_epoch), device=device)

    T = get_time_schedule(args, device)
    pos_coeff = Posterior_Coefficients(args, device)

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for source, _target in loader:
        # x = contrast1 = NCCT is the conditioning source; the target is never
        # read at inference. Matches test.py's second loop.
        source = source.to(device, non_blocking=True)
        x_t = torch.cat((torch.randn_like(source), source), axis=1)
        fake = sample_from_model(pos_coeff, gen, args.num_timesteps, x_t, T, args)

        arr = fake.detach().cpu().numpy()          # (B,1,H,W) in [-1,1]
        for i in range(arr.shape[0]):
            # undo LoadDataSet's (0,2,1) transpose
            np.save(args.out / names[written], arr[i, 0].T.astype(np.float32))
            written += 1
        print(f'  {written}/{len(names)}', end='\r', flush=True)

    print(f'\n[written] {written} slices -> {args.out}')
    print(f'  reassemble with: --slice_suffix _fake_B --in_range neg1_1')


if __name__ == '__main__':
    main()

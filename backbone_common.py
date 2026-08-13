#!/usr/bin/env python
"""
Shared plumbing for the two *segmentation* backbones we repurpose as NCCT->CECT
generators: MONAI's **SwinUNETR** and Beckschen's **TransUNet**.

Why this file exists
--------------------
ResViT, CyTran, CycleGAN and SynDiff each ship a complete training pipeline, so
adapting them is an I/O job. SwinUNETR and TransUNet do not: they are
architectures published with a *segmentation* trainer (Dice/CE on BTCV, Synapse,
BraTS) and a dataset loader for those benchmarks. Nothing in either repo maps
onto "predict a contrast-enhanced CT slice from a non-contrast one".

So for these two we keep the upstream *networks* untouched and supply the
training recipe ourselves — and we deliberately make that recipe **identical to
ResViT's**, so a difference in the final table is a difference in architecture
and not in optimisation:

    loss_G = lambda_adv * LSGAN(D(A, G(A)), 1) + lambda_l1 * L1(G(A), B)
    loss_D = 0.5 * lambda_adv * (LSGAN(D(A, G(A)), 0) + LSGAN(D(A, B), 1))

with a 3-layer conditional PatchGAN D (ndf=64), Adam(beta1=0.5), a constant lr
for `niter` epochs then linear decay to zero over `niter_decay`. Those are
exactly `ResViT/models/resvit_one.py` + `ResViT/options/*`. Setting
`--lambda_adv 0` drops the discriminator entirely and gives the pure supervised
L1 regression that the SwinUNETR/TransUNet papers themselves use — worth
running as an ablation, but not the default, because the rest of the benchmark
is adversarial.

Data
----
Both read the **pix2pix AB-PNG slices** that `prep_benchmark_data.py --format
pix2pix` already writes for ResViT/CyTran, so there is no third data format and
no extra disk copy: point `--dataroot` at `ResViT/datasets/vindr` and go.

Tensors are in **[-1, 1]** end to end (pix2pix convention), which is why
`reassemble_nifti.py` is called with `--in_range neg1_1` for both models.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

ARCHS = ('swinunetr', 'transunet')


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class ABSliceDataset(Dataset):
    """`<root>/<phase>/<case>_<zzzz>.png`, one image holding [A | B] side by side.

    A = NCCT (input), B = CECT (target); both single-channel, written by
    `prep_benchmark_data.py --format pix2pix`. Returns float tensors in [-1, 1]
    shaped (1, size, size), plus the file stem so inference can name its output
    `<case>_<zzzz>_fake_B.npy` and `reassemble_nifti.py` can find it again.

    No augmentation. Random flips would mirror left/right anatomy (liver ↔
    spleen), and random crops would break the 1:1 correspondence with the source
    grid that reassembly depends on — the slices are already exactly `size`.
    """

    def __init__(self, root: Path, phase: str, size: int = 256):
        self.dir = Path(root) / phase
        if not self.dir.is_dir():
            raise FileNotFoundError(
                f'{self.dir} not found — run `prep_benchmark_data.py --format pix2pix '
                f'--out {root}` first')
        self.paths: List[Path] = sorted(self.dir.glob('*.png'))
        if not self.paths:
            raise FileNotFoundError(f'no *.png in {self.dir}')
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        p = self.paths[i]
        im = np.asarray(Image.open(p).convert('L'), dtype=np.float32) / 255.0
        h, w = im.shape
        half = w // 2
        a, b = im[:, :half], im[:, half:]
        if a.shape[0] != self.size or a.shape[1] != self.size:
            raise ValueError(
                f'{p}: expected {self.size}x{self.size} per side, got {a.shape}. '
                f'Re-run prep with --size {self.size}.')
        to_t = lambda x: torch.from_numpy(x * 2.0 - 1.0).unsqueeze(0)   # [0,1] -> [-1,1]
        return to_t(a), to_t(b), p.stem


# --------------------------------------------------------------------------- #
# generators
# --------------------------------------------------------------------------- #
class _Tanh(nn.Module):
    """Squash a segmentation head's raw logits into the [-1, 1] image range.

    Both backbones end in a plain conv that was built to emit class logits. We
    keep that layer exactly as published and only bound its range, which is the
    smallest possible change that turns a segmenter into an image generator.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


def _build_swinunetr(size: int, in_ch: int, out_ch: int, feature_size: int) -> nn.Module:
    """MONAI SwinUNETR in 2-D.

    Two caveats worth carrying into the write-up:

    * `spatial_dims=2` is supported, but the published self-supervised
      `model_swinvit.pt` weights are **3-D only**, so the 2-D model trains from
      scratch. That is the honest state of this baseline — TransUNet gets an
      ImageNet-21k init and SwinUNETR does not.
    * MONAI moved `img_size` from required to deprecated to gone across 1.3→1.5,
      so try it and fall back.
    """
    try:
        from monai.networks.nets import SwinUNETR
    except ImportError as e:                                        # pragma: no cover
        raise SystemExit(
            'SwinUNETR needs MONAI:  pip install -r requirements_backbone.txt') from e

    kw = dict(in_channels=in_ch, out_channels=out_ch,
              feature_size=feature_size, spatial_dims=2)
    try:
        net = SwinUNETR(img_size=(size, size), **kw)
    except TypeError:
        net = SwinUNETR(**kw)          # MONAI >= 1.5 dropped img_size
    return _Tanh(net)


def _build_transunet(size: int, in_ch: int, out_ch: int,
                     transunet_dir: Path, vit_ckpt: Path | None) -> nn.Module:
    """Beckschen TransUNet (R50-ViT-B_16) as a 1->1 image generator.

    `in_ch` is ignored on purpose: `vit_seg_modeling.VisionTransformer.forward`
    already does `if x.size()[1] == 1: x = x.repeat(1, 3, 1, 1)`, because the
    R50 stem is a 3-channel ImageNet conv. Feeding it our single channel is the
    upstream-intended path, not a hack.

    `n_classes` becomes our output channel count, so the segmentation head emits
    one continuous map instead of class logits.
    """
    d = Path(transunet_dir)
    if not (d / 'networks' / 'vit_seg_modeling.py').is_file():
        raise SystemExit(f'TransUNet tree not found at {d}')
    sys.path.insert(0, str(d))
    from networks.vit_seg_modeling import VisionTransformer, CONFIGS  # noqa: E402

    cfg = CONFIGS['R50-ViT-B_16']
    cfg.n_classes = out_ch
    cfg.n_skip = 3
    # Embeddings computes patch_size = img_size // 16 // grid, so the grid must
    # track img_size to keep a 16x16 patch. At size=256 this is (16, 16).
    cfg.patches.grid = (size // 16, size // 16)

    net = VisionTransformer(cfg, img_size=size, num_classes=out_ch)
    if vit_ckpt is not None:
        vit_ckpt = Path(vit_ckpt)
        if not vit_ckpt.is_file():
            raise SystemExit(
                f'ViT checkpoint not found: {vit_ckpt}\n'
                f'  It is the same R50+ViT-B_16.npz that run_resvit.sh downloads;\n'
                f'  run `./run_resvit.sh setup`, or pass --vit_ckpt "" to train from scratch.')
        net.load_from(weights=np.load(vit_ckpt))
        print(f'  loaded ImageNet-21k init: {vit_ckpt}')
    else:
        print('  TransUNet: random init (no --vit_ckpt)')
    return _Tanh(net)


def build_generator(arch: str, size: int, in_ch: int = 1, out_ch: int = 1,
                    feature_size: int = 48,
                    transunet_dir: Path | None = None,
                    vit_ckpt: Path | None = None) -> nn.Module:
    if arch == 'swinunetr':
        return _build_swinunetr(size, in_ch, out_ch, feature_size)
    if arch == 'transunet':
        if transunet_dir is None:
            raise SystemExit('--transunet_dir is required for arch=transunet')
        return _build_transunet(size, in_ch, out_ch, transunet_dir, vit_ckpt)
    raise SystemExit(f'unknown --arch {arch!r} (choose from {ARCHS})')


# --------------------------------------------------------------------------- #
# discriminator + losses  (ports of ResViT/pix2pix, so the recipe matches)
# --------------------------------------------------------------------------- #
class NLayerDiscriminator(nn.Module):
    """70x70 conditional PatchGAN — `which_model_netD='basic'`, n_layers_D=3.

    Sees `cat(A, B)` on the channel axis, so `in_ch` is 2 for our 1->1 task.
    """

    def __init__(self, in_ch: int = 2, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        kw, pad = 4, 1
        seq: List[nn.Module] = [
            nn.Conv2d(in_ch, ndf, kernel_size=kw, stride=2, padding=pad),
            nn.LeakyReLU(0.2, True),
        ]
        mult = 1
        for n in range(1, n_layers + 1):
            prev, mult = mult, min(2 ** n, 8)
            stride = 2 if n < n_layers else 1
            seq += [
                nn.Conv2d(ndf * prev, ndf * mult, kernel_size=kw, stride=stride,
                          padding=pad, bias=False),
                nn.BatchNorm2d(ndf * mult),
                nn.LeakyReLU(0.2, True),
            ]
        seq += [nn.Conv2d(ndf * mult, 1, kernel_size=kw, stride=1, padding=pad)]
        self.model = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class LSGANLoss(nn.Module):
    """Least-squares GAN objective — ResViT's default (`--no_lsgan` off)."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.mse(pred, target)


def lambda_rule(niter: int, niter_decay: int):
    """pix2pix schedule: flat for `niter` epochs, then linear decay to 0."""
    def rule(epoch: int) -> float:
        return 1.0 - max(0, epoch - niter) / float(niter_decay + 1)
    return rule


def init_weights(net: nn.Module, gain: float = 0.02) -> nn.Module:
    """Normal(0, 0.02) init on conv/BN — pix2pix's `normal` init.

    Applied to the discriminator and to SwinUNETR. **Not** applied to TransUNet
    when a ViT checkpoint was loaded, since that would erase the pretrained
    weights we just read.
    """
    def fn(m: nn.Module) -> None:
        cn = m.__class__.__name__
        if hasattr(m, 'weight') and ('Conv' in cn or 'Linear' in cn):
            nn.init.normal_(m.weight.data, 0.0, gain)
            if getattr(m, 'bias', None) is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in cn:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(fn)
    return net

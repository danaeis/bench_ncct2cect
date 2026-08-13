#!/usr/bin/env python
"""
Train SwinUNETR or TransUNet as an NCCT->CECT generator on our VinDr slices.

Reads the pix2pix AB-PNG slices from `prep_benchmark_data.py --format pix2pix`
(the same directory ResViT and CyTran use) and trains with ResViT's exact
recipe — conditional PatchGAN + L1 — so the benchmark compares architectures
rather than optimisers. See backbone_common.py for why.

    python train_backbone.py --arch swinunetr \
        --dataroot ResViT/datasets/vindr --name vindr_swinunetr \
        --niter 25 --niter_decay 25 --batch 4

Checkpoints land in `<checkpoints_dir>/<name>/`:
    latest_net_G.pth        overwritten every epoch (what infer_backbone loads)
    <epoch>_net_G.pth       every --save_epoch_freq epochs
    latest_net_D.pth        so an interrupted run can resume adversarially
    log.txt                 per-epoch train losses + val L1/PSNR
Set --lambda_adv 0 for the non-adversarial (pure L1) ablation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backbone_common import (ARCHS, ABSliceDataset, LSGANLoss,
                             NLayerDiscriminator, build_generator, init_weights,
                             lambda_rule)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR between two [-1,1] arrays, on the [0,1] scale so it is comparable
    with what benchmark.py reports."""
    a01, b01 = (a + 1) / 2, (b + 1) / 2
    mse = float(np.mean((a01 - b01) ** 2))
    return float('inf') if mse == 0 else float(10 * np.log10(1.0 / mse))


@torch.no_grad()
def validate(netG: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    netG.eval()
    l1s, psnrs = [], []
    for A, B, _ in loader:
        A, B = A.to(device), B.to(device)
        fake = netG(A)
        l1s.append(float(torch.abs(fake - B).mean()))
        psnrs.append(_psnr(fake.cpu().numpy(), B.cpu().numpy()))
    netG.train()
    return float(np.mean(l1s)), float(np.mean(psnrs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arch', choices=ARCHS, required=True)
    ap.add_argument('--dataroot', type=Path, required=True,
                    help='dir holding train/ val/ test/ of AB-PNG slices')
    ap.add_argument('--name', required=True, help='experiment name')
    ap.add_argument('--checkpoints_dir', type=Path, default=Path('checkpoints'))
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--niter', type=int, default=25, help='epochs at full lr')
    ap.add_argument('--niter_decay', type=int, default=25, help='epochs of linear decay')
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--beta1', type=float, default=0.5)
    ap.add_argument('--lambda_l1', type=float, default=100.0, help="ResViT's lambda_A")
    ap.add_argument('--lambda_adv', type=float, default=1.0,
                    help='0 disables the discriminator entirely (pure L1 ablation)')
    ap.add_argument('--save_epoch_freq', type=int, default=5)
    ap.add_argument('--print_freq', type=int, default=200, help='iterations between loss lines')
    ap.add_argument('--feature_size', type=int, default=48, help='SwinUNETR width (mult of 12)')
    ap.add_argument('--transunet_dir', type=Path, default=Path('TransUNet'))
    ap.add_argument('--vit_ckpt', default='ResViT/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz',
                    help='TransUNet ImageNet-21k init; pass "" to train from scratch')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--continue_train', action='store_true')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu'
                          else 'cpu')
    if str(device) != args.device:
        print(f'  WARNING: CUDA unavailable, falling back to {device}')

    out = args.checkpoints_dir / args.name
    out.mkdir(parents=True, exist_ok=True)
    log = (out / 'log.txt').open('a')

    def say(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()

    # ---- data ---------------------------------------------------------------
    train_ds = ABSliceDataset(args.dataroot, 'train', args.size)
    val_ds = ABSliceDataset(args.dataroot, 'val', args.size)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, drop_last=True, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    say(f'[data] train {len(train_ds)} slices | val {len(val_ds)} slices '
        f'| {len(train_ld)} iters/epoch @ batch {args.batch}')

    # ---- models -------------------------------------------------------------
    vit_ckpt = Path(args.vit_ckpt) if (args.arch == 'transunet' and args.vit_ckpt) else None
    netG = build_generator(args.arch, args.size, 1, 1, args.feature_size,
                           args.transunet_dir, vit_ckpt).to(device)
    # Re-initialising TransUNet would throw away the checkpoint we just loaded.
    if args.arch == 'swinunetr':
        init_weights(netG)

    use_gan = args.lambda_adv > 0
    netD = init_weights(NLayerDiscriminator(in_ch=2)).to(device) if use_gan else None
    say(f'[model] {args.arch} | G params {sum(p.numel() for p in netG.parameters())/1e6:.1f}M'
        + (f" | D params {sum(p.numel() for p in netD.parameters())/1e6:.1f}M" if use_gan
           else ' | adversarial loss DISABLED (lambda_adv=0)'))

    optG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    schG = torch.optim.lr_scheduler.LambdaLR(optG, lambda_rule(args.niter, args.niter_decay))
    if use_gan:
        optD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        schD = torch.optim.lr_scheduler.LambdaLR(optD, lambda_rule(args.niter, args.niter_decay))

    start_epoch = 0
    if args.continue_train and (out / 'latest_net_G.pth').is_file():
        netG.load_state_dict(torch.load(out / 'latest_net_G.pth', map_location=device))
        if use_gan and (out / 'latest_net_D.pth').is_file():
            netD.load_state_dict(torch.load(out / 'latest_net_D.pth', map_location=device))
        if (out / 'latest_state.pth').is_file():
            start_epoch = int(torch.load(out / 'latest_state.pth')['epoch']) + 1
        for _ in range(start_epoch):
            schG.step()
            if use_gan:
                schD.step()
        say(f'[resume] from epoch {start_epoch}')

    crit_l1 = nn.L1Loss()
    crit_gan = LSGANLoss().to(device)
    total_epochs = args.niter + args.niter_decay

    # ---- train --------------------------------------------------------------
    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        run = {'G_L1': 0.0, 'G_GAN': 0.0, 'D': 0.0}
        for it, (A, B, _) in enumerate(train_ld):
            A, B = A.to(device, non_blocking=True), B.to(device, non_blocking=True)
            fake = netG(A)

            if use_gan:
                # D: real pair -> 1, fake pair -> 0 (detached), averaged.
                optD.zero_grad(set_to_none=True)
                loss_D = 0.5 * args.lambda_adv * (
                    crit_gan(netD(torch.cat((A, fake.detach()), 1)), False)
                    + crit_gan(netD(torch.cat((A, B), 1)), True))
                loss_D.backward()
                optD.step()
                run['D'] += loss_D.item()

            optG.zero_grad(set_to_none=True)
            loss_l1 = crit_l1(fake, B) * args.lambda_l1
            loss_G = loss_l1
            if use_gan:
                loss_adv = crit_gan(netD(torch.cat((A, fake), 1)), True) * args.lambda_adv
                loss_G = loss_G + loss_adv
                run['G_GAN'] += loss_adv.item()
            loss_G.backward()
            optG.step()
            run['G_L1'] += loss_l1.item()

            if args.print_freq and it % args.print_freq == 0:
                say(f'  e{epoch:03d} i{it:05d}/{len(train_ld)}  '
                    f'G_L1 {loss_l1.item():.4f}  '
                    + (f'G_GAN {loss_adv.item():.4f}  D {loss_D.item():.4f}' if use_gan else ''))

        schG.step()
        if use_gan:
            schD.step()

        n = max(1, len(train_ld))
        val_l1, val_psnr = validate(netG, val_ld, device)
        say(f'[epoch {epoch:03d}/{total_epochs - 1}] '
            f'G_L1 {run["G_L1"]/n:.4f} G_GAN {run["G_GAN"]/n:.4f} D {run["D"]/n:.4f} | '
            f'val_L1 {val_l1:.4f} val_PSNR {val_psnr:.2f} dB | '
            f'lr {schG.get_last_lr()[0]:.2e} | {time.time()-t0:.0f}s')

        torch.save(netG.state_dict(), out / 'latest_net_G.pth')
        torch.save({'epoch': epoch}, out / 'latest_state.pth')
        if use_gan:
            torch.save(netD.state_dict(), out / 'latest_net_D.pth')
        if (epoch + 1) % args.save_epoch_freq == 0:
            torch.save(netG.state_dict(), out / f'{epoch}_net_G.pth')

    say(f'[done] weights -> {out}/latest_net_G.pth')
    log.close()


if __name__ == '__main__':
    main()

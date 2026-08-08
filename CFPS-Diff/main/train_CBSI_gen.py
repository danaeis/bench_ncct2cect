import os
import time
import torch
import argparse
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from Networks_UNet_DDPM import Unet_class
from Nii_utils import setup_seed, Save_Parameter
from Networks_DDPM_trainer import GaussianDiffusion, DDPM_Trainer
from Networks_model_diffusion import Model_Validation_multitask
from Dataset_gen import CA_Dataset_harmonize_2D, CA_Dataset_harmonize
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*To copy construct from a tensor.*")



def main(args):
    train_writer = SummaryWriter(os.path.join(args.save_dir, 'log/train'), flush_secs=2)
    val_writer = SummaryWriter(os.path.join(args.save_dir, 'log/val'), flush_secs=2)
    print(args.save_dir)

    # ======== Initialize model and trainer ========
    if args.net.lower() == 'unet_ddpm_64_class':
        net = Unet_class(dim=64, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=1, condition_channels=args.cc,
                   condition=True, class_cond=args.class_cond, resnet_block_groups=8, seg_head=args.seg).cuda()
    elif args.net.lower() == 'unet_ddpm_32_class':
        net = Unet_class(dim=32, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=1, condition_channels=args.cc,
                   condition=True, class_cond=args.class_cond, resnet_block_groups=8, seg_head=args.seg).cuda()
    diffusion_model = GaussianDiffusion(net, image_size=args.ImageSize, timesteps=1000, loss_type=args.loss, seg_weight=args.seg_weight,
                                        objective=args.objective, activate=args.activate).cuda()
    trainer = DDPM_Trainer(diffusion_model, train_batch_size=args.bs, train_lr=args.lr_max)

    # ======== Learning rate scheduler setup ========
    scheduler_lr = MultiStepLR(trainer.opt, milestones=[int(0.5 * args.max_epoch), int(0.8 * args.max_epoch)], gamma=0.1, last_epoch=-1)

    # ======== Load dataset and dataloader ========
    root_dir = './Glioma_DATA/Preprocessing_DATA/Train'
    val_dir = './Glioma_DATA/Preprocessing_DATA/Val'
    train_set =  CA_Dataset_harmonize_2D(args, root_dir=root_dir)  # get training options
    val_set =  CA_Dataset_harmonize(args, root_dir=val_dir)  # get validation options
    train_dataloader = DataLoader(dataset=train_set, batch_size=args.bs, shuffle=True, num_workers=args.num_threads, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_set, batch_size=args.val_bs, shuffle=False, num_workers=args.num_threads, pin_memory=True)

    best_MAE = 1000
    Save_Parameter(args)  # save the training parameter
    for epoch in range(0, args.max_epoch):
        train_start_time = time.time()
        for param_group in trainer.opt.param_groups:
            lr = param_group['lr']
            break
        epoch_train_loss = []
        for i, DATA in enumerate(train_dataloader):
            x, x_cond = DATA['B'].cuda(), DATA['A'].cuda()  # x is target (T1Gd) and x_cond is source (T1 and T2-FLAIR)
            label = DATA['label'].cuda()  # label
            mask = DATA['ROI'].cuda()
            trainer.opt.zero_grad()
            trainer.model.train()
            loss = trainer.calculate_loss_multitask(x, x_cond, mask, seg_cond=None, label=label)
            loss_back = loss / args.gae
            loss_back.backward()
            if (i + 1) % args.gae == 0:
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1)
                trainer.opt.step()
            epoch_train_loss.append(loss.item())
            if i % 100 == 0:
                print('[%d/%d, %5d/%d] train_loss: %.3f ' % (epoch + 1, args.max_epoch, i + 1, len(train_dataloader), loss.item()))
        epoch_train_loss = np.mean(epoch_train_loss)
        scheduler_lr.step()
        print('[%d/%d] train_loss: %.3f' % (epoch + 1, args.max_epoch, epoch_train_loss))
        train_writer.add_scalar('loss', epoch_train_loss, epoch)
        train_writer.add_scalar('lr', lr, epoch)
        train_writer.add_scalar('time', (time.time() - train_start_time) / 60, epoch)
        print('Train One Epoch Time Taken: %.1f min' % ((time.time() - train_start_time) / 60))


        first_stage = (epoch <= args.max_epoch // 3 and epoch % 10 == 0)
        second_stage = (epoch > args.max_epoch // 3 and epoch <= args.max_epoch // 3 * 2 and epoch % 5 == 0)
        third_stage = (epoch > args.max_epoch // 3 * 2 and epoch % 5 == 0)

        if first_stage + second_stage + third_stage:
            print('epoch:', epoch)
            ref_timestep = max(10 * first_stage + 50 * second_stage + 100 * third_stage, 10)
            mMAE_val, mROI_MAE_val, _, _, mDICE_val, Metric_df_val = Model_Validation_multitask(args, epoch, trainer, {'val':val_dataloader}, dataset='val', save_dir=args.save_dir, writer={'val':val_writer}, ref_timestep=ref_timestep, inf_bs=18, train=True, save_image=False)
            if mMAE_val < best_MAE and epoch > 0:
                best_MAE = mMAE_val
                trainer.save(os.path.join(args.save_dir, 'best_MAE.pth'))
    trainer.save(os.path.join(args.save_dir, 'epoch' + str(epoch + 1) + '.pth'))
    train_writer.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # -------------------- Training settings
    parser.add_argument('--gpu', default='2', type=str, help='which gpu is used')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--max_epoch', default=100, type=int, help='all_epochs')
    parser.add_argument('--lr_max', default=0.00001, type=float, help='0.0001/0.00001')
    parser.add_argument('--gae', default=1, type=int, help='gradient_accumulate_every')
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    parser.add_argument('--val_bs', default=1, type=int, help='batch size')
    parser.add_argument('--num_threads', default=1, type=int, help='# threads for loading dataset')

    # -------------------- Data settings
    parser.add_argument('--data_dim', default='2D', type=str)
    parser.add_argument('--ImageSize', default=424, type=int, help='Spatial dimension cropped to 424 * 424')
    parser.add_argument('--MR_max', default=255, type=int, help='MR_max')
    parser.add_argument('--MR_min', default=0, type=int, help='MR_min')

    # -------------------- Model settings
    parser.add_argument('--net', default='unet_ddpm_32_class', type=str)
    parser.add_argument('--seg', default=True, type=bool, help='Introduce the auxiliary segmentation task')
    parser.add_argument('--objective', default='pred_x0', type=str, help='The output target of the generator', choices=['pred_x0', 'pred_noise'])
    parser.add_argument('--activate', default='tanh', type=str, help='tanh can only be used when pred_x0', choices=['none', 'tanh'])
    parser.add_argument('--cc', default=2, type=int, help='condition_channels: non-contrast MR (T1 and T2-FLAIR)')
    parser.add_argument('--class_cond', default=2, type=int, help='ET label types (enhancing and non-enhancing)')
    parser.add_argument('--output_nc', default=1, type=int, help='output T1Gd image only has one channel')

    # -------------------- Loss function
    parser.add_argument('--loss', default='l1', type=str, choices=['l1', 'l2', 'mix'])
    parser.add_argument('--seg_weight', default=0.1, type=float, help='weight of segmentation loss')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(args.seed)


    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    save_name = 'bs{}_epoch{}_gae{}_seed{}_class_seg'.format(args.bs, args.max_epoch, args.gae, args.seed)
    args.save_dir = os.path.join(
        'trained_models/CBSI_gen/{}_{}_{}_condition_act_{}__{}_{}'.format(args.objective, args.net, args.loss,
                                                                          args.activate, save_name, current_time))
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)

import os
import math
import argparse
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from Nii_utils import *
from Networks_ide import EfficientNet
from Dataset_ide import ThreeFolderDataset_2D_t1_t2f_t1c, test_pred_2D



def save_checkpoint(model, ckpt_dir, name):
    """Save model weights to the checkpoint directory
    Handles both single-GPU and multi-GPU (DataParallel) models"""
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), os.path.join(ckpt_dir, name))
    else:
        torch.save(model.state_dict(), os.path.join(ckpt_dir, name))



def main(opt):
    # -------------- Experiment naming & directory setup --------------
    opt.checkpoints_name = '%s_%s_%s' % (opt.model, opt.describe, current_time)
    opt.checkpoints_dir = join(opt.code_dir, 'checkpoints', opt.checkpoints_name)
    os.makedirs(opt.checkpoints_dir, exist_ok=True)
    opt.model_results = join(opt.checkpoints_dir, 'model_results')
    # Define a subfolder for logging and saving checkpoints
    save_name = 'bs{}_ImageSize{}_epoch{}_seed{}'.format(opt.batch_size, opt.ImageSize, opt.epoch, opt.seed)
    save_dir = join(opt.checkpoints_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    # Paths for logging and metadata
    opt.file_name_txt = join(save_dir, 'train_message.txt')
    train_writer = SummaryWriter(join(save_dir, 'log/train'), flush_secs=2)
    val_writer = SummaryWriter(join(save_dir, 'log/val'), flush_secs=2)

    # -------------- GPU Setup ----------------
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    setup_seed(opt.seed)
    Save_Parameter(opt)
    opt.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    # -------------- Identification Model Setup ----------------
    net = EfficientNet.from_name(model_name=f'efficientnet-{opt.model[-2:]}', in_channels=opt.inchannel, num_classes=opt.classes, dropout_rate=opt.drop).cuda()
    # Load pretrained weights for EfficientNet
    net_weights = net.state_dict()
    pre_weights = torch.load(f'./models/pretrain/efficientnet-b0-355c32eb.pth')  # Download from the official website
    pre_dict = {k: v for k, v in pre_weights.items()
                if net_weights[k].numel() == v.numel()}
    net.load_state_dict(pre_dict, strict=False)
    print(f'The model is 2D {opt.model}')

    # ----------------------loss & optimizer------------------------
    criterion = nn.BCEWithLogitsLoss().to(opt.device)  # Sigmoid-BCELoss
    optimizer = optim.Adam(net.parameters(), lr=opt.lr_max, weight_decay=5e-4)

    # -------------- Learning Rate Scheduler ----------------
    if opt.warmup:
        # warm_up_with_cosine_lr
        int_decay = opt.lr_min / opt.lr_max
        zoom = 1 - int_decay

        warm_up_with_cosine_lr = lambda \
            epoch: int_decay + epoch / opt.warm_up_epochs * zoom if epoch <= opt.warm_up_epochs else int_decay + zoom * 0.5 * (
                    math.cos((epoch - opt.warm_up_epochs) / (opt.epoch - opt.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    else:
        scheduler = MultiStepLR(optimizer, milestones=[int((1 / 3) * opt.epoch), int((2 / 3) * opt.epoch)], gamma=0.1, last_epoch=-1)

    # -------------- Dataset Setup ----------------
    cv = "./glioma_data/train_val_fold.txt"
    ROI_train_dir = './glioma_data/Train_val'
    torch.set_printoptions(precision=4, sci_mode=False)

    ckpt_dir = join(save_dir, 'train_model')
    os.makedirs(ckpt_dir, exist_ok=True)

    train_set = ThreeFolderDataset_2D_t1_t2f_t1c(ROI_train_dir, join(opt.image_dir, opt.sample_name, 'predictions_train_val'), join(opt.image_dir, opt.sample_name + '_reverse', 'predictions_train_val'), cv=cv, train=True, test=False)
    # val_set = ThreeFolderDataset_2D_t1_t2f_t1c(ROI_train_dir, join(opt.image_dir, opt.sample_name, 'predictions_train_val'), join(opt.image_dir, opt.sample_name + '_reverse', 'predictions_train_val'), cv=cv, train=False, test=False)

    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.batch_size, drop_last=False)
    # val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.batch_size, drop_last=False)
    print(f'Size of train_dataset:{len(train_set)}.')
    # print(f'Size of val_dataset:{len(val_set)}.')
    print('Data prepared.')

    # -------------- Metrics Init ----------------
    threshold = 0.5
    best_val_auc, best_epoch = 0, 0

    # ==================== Training Loop ====================
    print('Start training.')
    for epoch in range(opt.start_epoch, opt.epoch):
        net.train()
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break
        train_acc_list = []
        train_loss_list = []
        y_true = torch.tensor([]).cuda()
        y_pred = torch.tensor([]).cuda()
        y_binary = torch.tensor([]).cuda()
        # ------------------ Batch Training ------------------
        for i, (image, label) in enumerate(train_loader):
            image = image.cuda()
            label = label.cuda()
            net.zero_grad()
            y = net(image)
            loss = criterion(y, label)
            # Optional flood loss
            if opt.do_flood:
                flood_loss = (loss - opt.flood).abs() + opt.flood
                flood_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            # Metrics logging
            y = torch.sigmoid(y)
            y_binary = torch.cat([y_binary, (y.detach() > threshold)])
            y_true = torch.cat([y_true, label.detach()])
            y_pred = torch.cat([y_pred, y.detach()])

            hit = ((y.detach() < threshold) ^ label.bool()).sum()
            train_acc_list.append(np.array(hit.cpu()))
            train_loss_list.append(np.array(loss.detach().cpu()))

        scheduler.step()
        # Calculate epoch-level metrics
        train_loss = np.array(train_loss_list).mean()
        train_acc = np.array(train_acc_list).sum() / len(train_set)
        train_auc = roc_auc_score(y_true.cpu(), y_pred.cpu())

        train_precision, train_recall, train_F1_score, _ = precision_recall_fscore_support(y_true.int().cpu(),
                                                                                           y_binary.int().cpu(),
                                                                                           average='binary')
        # TensorBoard Logging
        train_writer.add_scalar('lr', lr, epoch)
        train_writer.add_scalar('loss', train_loss, epoch)
        train_writer.add_scalar('AUC', train_auc, epoch)
        train_writer.add_scalar('ACC', train_acc, epoch)
        train_writer.add_scalar('F1_score', train_F1_score, epoch)
        # train_writer.add_scalar('recall', train_recall, epoch)
        # train_writer.add_scalar('precision', train_precision, epoch)

        pred_all_val = {'ID': [], "pred0": [], "pred1": [], "label0": [], "label1": [], "label": [], "softmax_pred": []}
        # ------------------ Validation Phase ------------------
        net.eval()
        val_loss_list = []
        val_acc_list = []
        y_true = torch.tensor([]).cuda()
        y_pred = torch.tensor([]).cuda()
        y_binary = torch.tensor([]).cuda()

        # Define prediction directories for forward and reverse inputs
        folder_b = join(opt.image_dir, opt.sample_name, 'predictions_train_val')
        folder_c = join(opt.image_dir, opt.sample_name + '_reverse', 'predictions_train_val')

        # Load filenames from both classes
        files_a_0 = os.listdir(join(ROI_train_dir, 'ET-0'))
        files_a_1 = os.listdir(join(ROI_train_dir, 'ET-1'))

        # Load validation sample names
        with open(cv, 'r') as f:
            val_name_list = f.readlines()
            val_name_list = [i.strip('\n') for i in val_name_list]
            val_name_list = [i.split('\t')[opt.k] for i in val_name_list]

        # Filter the file list by validation samples
        files_a_0 = [i for i in files_a_0 if i in val_name_list]
        files_a_1 = [i for i in files_a_1 if i in val_name_list]
        files_a = files_a_0 + files_a_1
        labels_read = [0] * len(files_a_0) + [1] * len(files_a_1)
        with torch.no_grad():
            for i, name in enumerate(tqdm(files_a)):
                reverse = 0 if labels_read[i] == 1 else 1
                T1, _, _, _ = NiiDataRead(join(ROI_train_dir, f'flair{labels_read[i]}_after/{name}/T1.nii.gz'))
                T2F, _, _, _ = NiiDataRead(join(ROI_train_dir, f'flair{labels_read[i]}_after/{name}/T2F.nii.gz'))
                ROI, _, _, _ = NiiDataRead(join(ROI_train_dir, f'flair{labels_read[i]}_after/{name}/ROI.nii.gz'))
                Fake, _, _, _ = NiiDataRead(join(folder_b, f'label{labels_read[i]}_{name}.nii.gz'))
                Fake_reverse, _, _, _ = NiiDataRead(join(folder_c, f'label{reverse}_{name}.nii.gz'))

                # Get model predictions and labels
                preds, labels = test_pred_2D(net, {'t1': T1, 't2f': T2F}, Fake, Fake_reverse, ROI, opt)


                for j in range(len(preds)):
                    y = preds[j]
                    label = labels[j]
                    loss = criterion(y, label)
                    val_loss_list.append(np.array(loss.cpu()))
                    y_binary = torch.cat([y_binary, (y.detach() > threshold)])
                    y_true = torch.cat([y_true, label.detach()])
                    y_pred = torch.cat([y_pred, y.detach()])
                    hit = ((y.detach() < threshold) ^ label.bool()).sum()
                    val_acc_list.append(np.array(hit.cpu()))
                    if j == 0:
                        pred_all_val['ID'].append(name)
                    pred_all_val[f"pred{j}"].append(y.detach().cpu().numpy()[0])
                    pred_all_val[f"label{j}"].append(label.detach().cpu().numpy()[0])
                pred_all_val[f"label"].append(labels_read[i])

                # Compute final softmax-based prediction (binary-class weighted softmax)
                AAA = pred_all_val[f"label"][-1] * \
                      torch.softmax(torch.tensor([pred_all_val[f"pred0"][-1], pred_all_val[f"pred1"][-1]]), dim=0)[0]
                BBB = (1 - pred_all_val[f"label"][-1]) * \
                      torch.softmax(torch.tensor([pred_all_val[f"pred0"][-1], pred_all_val[f"pred1"][-1]]), dim=0)[-1]
                pred_all_val[f"softmax_pred"].append((AAA + BBB).item())

            # Compute evaluation metrics
            val_auc_softmax = roc_auc_score(list(pred_all_val[f"label"]), list(pred_all_val[f"softmax_pred"]))
            val_loss = np.array(val_loss_list).mean()
            val_acc = np.array(val_acc_list).sum() / len(val_acc_list)
            val_auc = roc_auc_score(y_true.cpu(), y_pred.cpu())
            val_precision, val_recall, val_F1_score, _ = precision_recall_fscore_support(y_true.int().cpu(),
                                                                                            y_binary.int().cpu(),
                                                                                            average='binary')
            # Log metrics
            val_writer.add_scalar('loss', val_loss, epoch)
            val_writer.add_scalar('AUC', val_auc, epoch)
            val_writer.add_scalar('softmax_auc', val_auc_softmax, epoch)
            val_writer.add_scalar('ACC', val_acc, epoch)
            val_writer.add_scalar('F1_score', val_F1_score, epoch)

        # Save best model if AUC improved after 10 epochs
        if val_auc > best_val_auc and epoch >= 10:
            save_checkpoint(net, ckpt_dir, f'best_auc.pth')
            best_val_auc = val_auc
        save_checkpoint(net, ckpt_dir, f'last.pth')



if __name__ == '__main__':
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    parser = argparse.ArgumentParser()
    # =================== Basic parameters ===================
    parser.add_argument('--gpu_ids', type=str, default='1', help='which gpu is used')
    parser.add_argument('--k', type=int, default=0, help='fold')
    parser.add_argument('--epoch', type=int, default=100, help='all_epochs')
    parser.add_argument('--model', type=str, default='EfficientNet_b0')
    parser.add_argument('--describe', type=str, default='CBSI_ide', help='t1_t2f| t1 | t2f |t1c')
    parser.add_argument('--start_epoch', type=int, default=0, help='all_epochs')
    # set base_options
    # =================== Parameter adjustment parameter ===================
    parser.add_argument('--inchannel', type=int, default=3, help='input channel (T1, T2-FLAIR, synthetic T1Gd)')
    parser.add_argument('--classes', type=int, default=1, help='random seed')
    parser.add_argument('--lr_max', type=float, default=5e-4, help='top learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-5, help='initial learning rate')
    parser.add_argument('--warm_up_epochs', type=int, default=5, help='warm_up_epochs')
    parser.add_argument('--label_smooth', type=bool, default=False, help='label_smooth')
    # set base_options
    parser.add_argument('--code_dir', default='/home/zky/Classify', required=False, help='code_dir')
    parser.add_argument("--image_dir", type=str, default='/home/zky/Diffusion_train_mask/trained_models/DDPM/pred_x0_unet_ddpm_64_class_l1_condition_act_tanh_pretrain_True_bs3_epoch60_gae1_seed42_class_seg_Jun18_12-23-18/', help="name of the dataset")
    parser.add_argument('--sample_name', type=str, default='best_MAE_ddim_100', help='best_MAE_ddim_100 |epoch200_ddim_100')
    parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--ImageSize', type=int, default=424, help='then crop to this size')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--do_flood', type=bool, default=True, help='do flood loss')
    parser.add_argument('--flood', type=float, default=0.1, help='flood loss')
    # parser.add_argument('--warmup', type=bool, default=True)
    parser.add_argument('--warmup', action='store_false')
    # set train_options
    parser.add_argument('--isTrain', action='store_false', help='isTrain')
    parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
    parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--drop', type=float, default=0.2, help='dropout rate 0~1 ')
    # set val_options
    parser.add_argument('--disx', type=int, default=10120, help='frequency of showing training results on console')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids

    main(opt)



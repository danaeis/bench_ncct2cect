import os
import sys
import math
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from Networks_ide import EfficientNet
from sklearn.metrics import accuracy_score, recall_score
sys.path.append('..')

from Nii_utils import setup_seed, NiiDataRead, harmonize_mr, find_best_threshold





def Test_Pred_2D(net, Real, Syn_T1C, opt):
    preds = {'en': torch.zeros(1).cuda(), 'non': torch.zeros(1).cuda()}
    # Data preprocessing
    if isinstance(Real, dict):
        for image_ID in [k for k in Real.keys() if 'roi' not in k]:
            Real[image_ID] = harmonize_mr(Real[image_ID])
    else:
        Real = harmonize_mr(Real)
    Syn_T1C = {k: harmonize_mr(v) for k, v in Syn_T1C.items()}

    # Extract slices and construct an image list
    slices = np.unique(Real['roi'].nonzero()[0])
    image_list_en, image_list_non = [], []
    for slice in slices:
        data_en = np.stack((Real['t1'][slice], Real['t2f'][slice], Syn_T1C['en'][slice]), axis=0)
        data_non = np.stack((Real['t1'][slice], Real['t2f'][slice], Syn_T1C['non'][slice]), axis=0)
        image_list_en.append(data_en)
        image_list_non.append(data_non)

    # Build an image dictionary
    Image = {'en': np.array(image_list_en), 'non': np.array(image_list_non)}
    count = len(slices)

    # Batch processing images
    for ET in ['en', 'non']:
        n_num = math.ceil(Image[ET].shape[0] / opt.batch_size)
        for n in range(n_num):
            start = n * opt.batch_size
            end = start + opt.batch_size if n < n_num - 1 else Image[ET].shape[0]
            one_image = torch.from_numpy(Image[ET][start:end]).float().cuda()
            y = torch.sigmoid(net(one_image))
            k = (y >= 0.5).float()
            preds[ET] += torch.sum(k.detach(), dim=0)
    # normalization
    preds['en'] /= count
    preds['non'] /= count
    return preds


def main(opt):
    # ----------------------Classification Model------------------------
    model_dir = './...'  # The parameters path of the well-traind identification model of CBSI need to be written
    net = EfficientNet.from_name(model_name=f'efficientnet-{opt.model[-2:]}', in_channels=opt.inchannel, num_classes=opt.classes).cuda()
    net.load_state_dict(torch.load(join(model_dir, 'train_model', 'best_auc.pth')), strict=True)
    print(f'The model is 2D {opt.model}')

    # ------------------Testing------------------
    if not os.path.exists(join(opt.save_dir, 'pred_all_CBSI.csv')):
        print('!!!!!!!!Testing stage start!!!!!!!!!!')
        pred_all = {'ID': [], "softmax_pred": [], "argmax_pred": [], "label": []}
        ID_label_dict = {}
        for i, (ID, label) in enumerate(tqdm(zip(opt.labels_read['ID'].tolist(), opt.labels_read['ET_label'].tolist()))):
            if 'brats' in ID.lower():
                ID_label_dict[ID] = label

        net.eval()
        with torch.no_grad():
            for ID in tqdm(ID_label_dict.keys()):
                # Load the data and perform preprocessing
                T1, T2F, ROI, BrainMask = [NiiDataRead(join(opt.ROI_dir, f'{ID}/{modality}.nii.gz'))[0] for modality in ['T1', 'T2F', 'ROI', 'Brain_mask']]
                Syn_T1C_en, Syn_T1C_non = [NiiDataRead(join(opt.pred_image_dir, f'predictions/label{label}_{ID}.nii.gz'))[0] for label in [1, 0]]

                # Cropping and Interpolation
                zz, _, _ = ROI.nonzero()
                z_min, z_max = zz.min(), zz.max()
                data = {k: v[z_min:z_max + 1] for k, v in
                        zip(['T1', 'T2F', 'ROI', 'BrainMask', 'Syn_T1C_en', 'Syn_T1C_non'],
                            [T1, T2F,  ROI, BrainMask, Syn_T1C_en, Syn_T1C_non])}
                if data['T1'].shape[1] != opt.ImageSize:
                    data = {k: F.interpolate(torch.tensor(v).unsqueeze(0).unsqueeze(0),
                                             size=(v.shape[0], opt.ImageSize, opt.ImageSize), mode='nearest')[0][0].numpy()
                            for k, v in data.items()}
                if opt.ImageSize != opt.CropSize:
                    crop_start = (opt.ImageSize - opt.CropSize) // 2
                    crop_end = crop_start + opt.CropSize
                    data = {k: v[:, crop_start:crop_end, crop_start:crop_end] for k, v in data.items()}
                data['ROI'][data['ROI'] != 0] = 1

                # Start Inference
                # Contrastive Maximum Learning of CBSI
                preds = Test_Pred_2D(net, {'t1': data['T1'], 't2f': data['T2F'], 'roi': data['ROI']},
                                     {'en': data['Syn_T1C_en'], 'non': data['Syn_T1C_non']}, opt)
                pred_all['ID'].append(ID)
                pred_all['argmax_pred'].append(torch.cat((preds['non'], preds['en']), dim=0).argmax().item())
                pred_all['softmax_pred'].append(torch.cat((preds['non'], preds['en']), dim=0).softmax(dim=0)[1].item())
                pred_all['label'].append(ID_label_dict[ID])
        # Save pred results
        pd.DataFrame(pred_all).to_csv(join(opt.save_dir, 'pred_all_CBSI.csv'), index=False)
    else:
        pred_all = pd.read_csv(join(opt.save_dir, 'pred_all_CBSI.csv'))
    # ------------------Calculation index------------------
    metrics_all = {'metrics': [], "auc": [], "acc": [], "Recall": [], "specificity": []}
    try:
        auc = roc_auc_score(pred_all[f"label"], pred_all[f"argmax_pred"])
    except:
        auc = 0
    acc = accuracy_score(pred_all[f"label"], pred_all[f"argmax_pred"])  # Accuracy
    recall = recall_score(pred_all[f"label"], pred_all[f"argmax_pred"], pos_label=1)  # Recall
    specificity = recall_score(pred_all[f"label"], pred_all[f"argmax_pred"], pos_label=0)
    print(f'Mode: argmax_pred, AUC: {auc}, Accuracy: {acc}, Recall: {recall}, Specificity: {specificity}')

    metrics_all['metrics'].append('argmax_pred')
    metrics_all['auc'].append(auc)
    metrics_all['acc'].append(acc)
    metrics_all['Recall'].append(recall)
    metrics_all['specificity'].append(specificity)

    for pred_mode in ['softmax_pred']:
        try:
            auc = roc_auc_score(pred_all[f"label"], pred_all[pred_mode])
            best_threshold = find_best_threshold(pred_all[f"label"], pred_all[pred_mode])
        except:
            auc = 0
            best_threshold = 0.5
        y_pred_binary = [int(p >= best_threshold) for p in pred_all[pred_mode]]
        acc = accuracy_score(pred_all[f"label"], y_pred_binary)
        recall = recall_score(pred_all[f"label"], y_pred_binary, pos_label=1)
        specificity = recall_score(pred_all[f"label"], y_pred_binary, pos_label=0)
        metrics_all['metrics'].append(pred_mode)
        metrics_all['auc'].append(auc)
        metrics_all['acc'].append(acc)
        metrics_all['Recall'].append(recall)
        metrics_all['specificity'].append(specificity)
        print(f'Mode: {pred_mode}, AUC: {auc}, Accuracy: {acc}, Recall: {recall}, Specificity: {specificity}')

    # Save metrics
    pd.DataFrame(pred_all).to_csv(join(opt.save_dir, 'pred_all.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='1', help='GPU IDs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--model', type=str, default='EfficientNet_b0', help='Model name')
    parser.add_argument('--inchannel', type=int, default=4, help='Input channels')
    parser.add_argument('--classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--ImageSize', type=int, default=424, help='Image size')  # freeze
    parser.add_argument('--CropSize', type=int, default=424, help='Crop size')  # freeze
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    opt.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    setup_seed(opt.seed)

    # ---------------------Dataset-----------------------
    # opt.ROI_dir = './glioma_data/MICCAI_2023/BraTS-GLI/'
    # opt.pred_image_dir = './glioma_data/MICCAI_2023/CBSI'
    # opt.labels_read = pd.read_csv("./glioma_data/MICCAI_2023/all_data_information.csv")
    # opt.save_dir = './glioma_data/MICCAI_2023/Comparison_model_BraTS-GLI/'


    opt.ROI_dir = './glioma_data/MICCAI_2023/BraTS-Africa/95_Glioma_harmonization'
    opt.pred_image_dir = './glioma_data/MICCAI_2023/CBSI/'
    opt.labels_read = pd.read_csv("./glioma_data/MICCAI_2023/BraTS-Africa/all_data_information.csv")
    opt.save_dir = './glioma_data/MICCAI_2023/Comparison_model_BraTS-Africa/'


    main(opt)

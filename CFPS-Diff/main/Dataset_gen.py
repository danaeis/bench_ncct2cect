import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from os.path import join
from torch.utils.data import Dataset, DataLoader
from albumentations import (ShiftScaleRotate, Compose, HorizontalFlip, VerticalFlip, Resize)

from Nii_utils import NiiDataRead, harmonize_mr


class CA_Dataset_harmonize_2D(Dataset):
    def __init__(self, opt, root_dir):
        self.image_labels = []
        self.image_filenames = []
        self.root_dir = root_dir
        self.a_t1 = []
        self.a_t2f = []
        self.b_t1c = []
        self.ROI = []
        self.Brain = []
        self.image_labels = []
        self.MR_min = opt.MR_min
        self.MR_max = opt.MR_max

        for c in os.listdir(root_dir):
            if c == 'ET-1':
                label = 1
            elif c == 'ET-0':
                label = 0
            class_dir = join(root_dir, c)
            for ID in tqdm(os.listdir(class_dir)):
                image_filenames = join(c, ID)
                a_t1, spacing, origin, direction = NiiDataRead(join(self.root_dir, image_filenames, 'T1.nii.gz'))
                a_t2f, _, _, _ = NiiDataRead(join(self.root_dir, image_filenames, 'T2F.nii.gz'))
                b_t1c, _, _, _ = NiiDataRead(join(self.root_dir, image_filenames, 'T1C.nii.gz'))
                ROI, _, _, _ = NiiDataRead(join(self.root_dir, image_filenames, 'ROI.nii.gz'))
                Brain, _, _, _ = NiiDataRead(join(self.root_dir, image_filenames, 'Brain_mask.nii.gz'))
                ROI[ROI > 0] = 1
                Brain[Brain > 0] = 1

                for slice in range(b_t1c.shape[0]):
                    self.a_t1.append(a_t1[slice])
                    self.a_t2f.append(a_t2f[slice])
                    self.b_t1c.append(b_t1c[slice])
                    self.ROI.append(ROI[slice])
                    self.Brain.append(Brain[slice])
                    self.image_labels.append(label)

        self.trans_train = Compose([ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2,
                                                     rotate_limit=180, p=0.5,
                                                     border_mode=cv2.BORDER_CONSTANT, value=0,
                                                     interpolation=cv2.INTER_NEAREST),
                                    HorizontalFlip(p=0.3), VerticalFlip(p=0.3)])
        self.all_patch_num = len(self.b_t1c)

    def __getitem__(self, index):
        label = self.image_labels[index]
        a_t1 = self.a_t1[index]
        a_t2f = self.a_t2f[index]
        b_t1c = self.b_t1c[index]
        ROI = self.ROI[index]
        Brain = self.Brain[index]
        a_t1 = torch.tensor(harmonize_mr(a_t1, self.MR_min, self.MR_max)).float()
        a_t2f = torch.tensor(harmonize_mr(a_t2f, self.MR_min, self.MR_max)).float()
        b_t1c = torch.tensor(harmonize_mr(b_t1c, self.MR_min, self.MR_max)).float()
        ROI = torch.tensor(ROI).float()
        Brain = torch.tensor(Brain).float()
        label = torch.tensor(label).long()

        # random Horizontal and Vertical Flip to both image and mask
        image_3 = np.concatenate((a_t1[:, :, np.newaxis], a_t2f[:, :, np.newaxis], b_t1c[:, :, np.newaxis]), axis=2)
        mask_2 = np.concatenate((ROI[:, :, np.newaxis], Brain[:, :, np.newaxis]), axis=2)


        augmented = self.trans_train(image=image_3, mask=mask_2)
        image = augmented['image']
        mask = augmented['mask']

        a_t1 = torch.tensor(image[:, :, 0]).unsqueeze(0)
        a_t2f = torch.tensor(image[:, :, 1]).unsqueeze(0)
        b_t1c = torch.tensor(image[:, :, 2]).unsqueeze(0)
        ROI = torch.tensor(mask[:, :, 0]).unsqueeze(0)
        Brain = torch.tensor(mask[:, :, 1]).unsqueeze(0)
        label = torch.tensor(label).long()

        return {
            'A': torch.cat((a_t1, a_t2f), dim=0),
            'B': b_t1c,
            'ROI': ROI,
            'Brain': Brain,
            'label': label
        }


    def __len__(self):
        return self.all_patch_num



class CA_Dataset_harmonize(Dataset):
    def __init__(self, opt, root_dir, train=False):
        self.image_labels = []
        self.image_filenames = []
        self.root_dir = root_dir
        self.a_t1 = []
        self.a_t2f = []
        self.b_t1c = []
        self.b_ROI = []
        self.b_brain = []
        self.name = []
        self.MR_min = opt.MR_min
        self.MR_max = opt.MR_max
        self.train = train

        for c in os.listdir(root_dir):
            if c == 'ET-1':
                label = 1
            elif c == 'ET-0':
                label = 0
            class_dir = join(root_dir, c)
            for ID in tqdm(os.listdir(class_dir)):
                image_filenames = join(c, ID)
                self.name.append(ID)
                self.a_t1.append(join(self.root_dir, image_filenames, 'T1.nii.gz'))
                self.a_t2f.append(join(self.root_dir, image_filenames, 'T2F.nii.gz'))
                self.b_t1c.append(join(self.root_dir, image_filenames, 'T1C.nii.gz'))
                self.b_ROI.append(join(self.root_dir, image_filenames, 'ROI.nii.gz'))
                self.b_brain.append(join(self.root_dir, image_filenames, 'Brain_mask.nii.gz'))
                self.image_labels.append(label)

        self.all_patch_num = len(self.b_t1c)

    def __getitem__(self, index):
        label = self.image_labels[index]
        a_t1, spacing, origin, direction = NiiDataRead(self.a_t1[index])
        a_t2f, _, _, _ = NiiDataRead(self.b_t1c[index])
        b_t1c, spacing, origin, direction = NiiDataRead(self.b_t1c[index])
        b_ROI, _, _, _ = NiiDataRead(self.b_ROI[index])
        b_brain, _, _, _ = NiiDataRead(self.b_brain[index])
        image_parameter = (spacing, origin, direction)
        name = self.name[index]
        b_ROI[b_ROI > 0] = 1
        b_brain[b_brain > 0] = 1

        a_t1 = harmonize_mr(a_t1, self.MR_min, self.MR_max)
        a_t2f = harmonize_mr(a_t2f, self.MR_min, self.MR_max)
        b_t1c = harmonize_mr(b_t1c, self.MR_min, self.MR_max)

        a_t1 = torch.tensor(a_t1).unsqueeze(1)
        a_t2f = torch.tensor(a_t2f).unsqueeze(1)
        b_t1c = torch.tensor(b_t1c)
        b_ROI = torch.tensor(b_ROI)
        b_brain = torch.tensor(b_brain)
        # label = torch.tensor(label).long()
        return {
            'A': torch.cat((a_t1, a_t2f), dim=1),
            'B': b_t1c,
            'tumor_mask': b_ROI,
            'brain_mask': b_brain,
            'label': label,
            'ID': name,
            'image_parameter': image_parameter,
        }

    def __len__(self):
        return len(self.image_labels)



if __name__ == '__main__':
    pass
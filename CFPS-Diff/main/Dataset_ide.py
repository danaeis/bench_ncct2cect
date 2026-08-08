import os
import torch
import numpy as np
from torch.utils.data import Dataset
from os.path import join
import SimpleITK as sitk
import torchvision.transforms as transforms
from torch.nn.functional import one_hot
import skimage.measure as measure
from util.Nii_utils import *
from dataset.dataset_CA import mask_decoder, random_crop_with_mask_three


class ThreeFolderDataset(Dataset):
    def __init__(self, folder_a, folder_b, folder_c, train, cv=None, test=False, fold=0):
        self.folder_a = folder_a
        self.folder_b = folder_b
        self.folder_c = folder_c
        self.files_b = sorted(os.listdir(folder_b))
        # assert len(self.files_b) == len(self.files_b) == len(self.files_c), "All folders must have same number of files"

        if not test:
            # 读取txt文件
            with open(cv, 'r') as f:
                delete_name_list = f.readlines()
                delete_name_list = [i.strip('\n') for i in delete_name_list]
                delete_name_list = [i.split('\t')[fold] for i in delete_name_list]

        self.file_list = []
        self.label_list = []
        for name in self.files_b:
            if train and name[7:-7] in delete_name_list:
                continue
            elif not train and not test and name[7:-7] not in delete_name_list:
                continue
            reverse = '0' if name[5] == '1' else '1'
            self.file_list.append([join(folder_a, f'flair{name[5]}_after', name[7:-7]), join(folder_b, f'{name[:-7]}.nii.gz')])
            self.label_list.append(1)
            self.file_list.append([join(folder_a, f'flair{name[5]}_after', name[7:-7]), join(folder_c, f'label{reverse}_{name[7:-7]}.nii.gz')])
            self.label_list.append(0)
        print('label1 : ',sum(self.label_list), 'label0 : ', len(self.label_list)-sum(self.label_list))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        T1, _, _, _ = NiiDataRead(join(self.file_list[idx][0], 'T1.nii.gz'))
        T2F, _, _, _ = NiiDataRead(join(self.file_list[idx][0], 'T2F.nii.gz'))
        # T2F, _, _, _ = NiiDataRead(join(self.file_list[idx][0], 'T1C.nii.gz'))
        synthsis, _, _, _ = NiiDataRead(self.file_list[idx][1])
        label = self.label_list[idx]

        T1 = harmonize_mr(T1)
        T2F = harmonize_mr(T2F)
        synthsis = harmonize_mr(synthsis)

        if T1.shape[0]-12 <= 0:
            z_start = [0]
        else:
            z_start = np.random.randint(0, T1.shape[0]-12, 1)
        T1 = T1[z_start[0]:z_start[0]+12, :, :]
        T2F = T2F[z_start[0]:z_start[0]+12, :, :]
        synthsis = synthsis[z_start[0]:z_start[0]+12, :, :]

        assert T1.shape[0] == 12
        assert T2F.shape[0] == 12
        assert synthsis.shape[0] == 12
        Real = torch.from_numpy(np.concatenate((T1[np.newaxis, ...], T2F[np.newaxis, ...]), 0)).float()
        synthsis = torch.from_numpy(synthsis)

        """if self.aug:
            # random Horizontal and Vertical Flip to both image and mask
            p1 = np.random.choice([0, 1])
            p2 = np.random.choice([0, 1])
            self.trans = transforms.Compose([
                transforms.RandomHorizontalFlip(p1),
                transforms.RandomVerticalFlip(p2),
                transforms.RandomRotation(10, resample=False, expand=False, center=None),
            ])
            Real = self.trans(Real)
            synthsis = self.trans(synthsis)"""
        # Convert data to PyTorch tensor
        # Real = Real.unsqueeze(0).float()
        synthsis = synthsis.unsqueeze(0).float()

        # if self.aug:
        #     # random Horizontal and Vertical Flip to both image and mask
        #     p1 = np.random.choice([0, 1])
        #     p2 = np.random.choice([0, 1])
        #     self.trans = transforms.Compose([
        #         transforms.RandomHorizontalFlip(p1),
        #         transforms.RandomVerticalFlip(p2),
        #         transforms.RandomRotation(10, resample=False, expand=False, center=None),
        #     ])
        #     Real = self.trans(Real)
        #     synthsis = self.trans(synthsis)
        # # Convert data to PyTorch tensor
        # Real = torch.from_numpy(Real).unsqueeze(0).float()
        # synthsis = torch.from_numpy(synthsis).unsqueeze(0).float()

        data = torch.cat((Real, synthsis), dim=0)
        label = torch.tensor(label).unsqueeze(0).float()
        return data, label



class ThreeFolderDataset_plus(Dataset):
    def __init__(self, folder_a, folder_b, folder_c, train, cv=None, test=False, fold=0):
        self.folder_a = folder_a
        self.folder_b = folder_b
        self.folder_c = folder_c
        self.files_b = sorted(os.listdir(folder_b))
        # assert len(self.files_b) == len(self.files_b) == len(self.files_c), "All folders must have same number of files"

        if not test:
            # 读取txt文件
            with open(cv, 'r') as f:
                delete_name_list = f.readlines()
                delete_name_list = [i.strip('\n') for i in delete_name_list]
                delete_name_list = [i.split('\t')[fold] for i in delete_name_list]

        self.file_list = []
        self.label_list = []
        for name in self.files_b:
            if train and name[7:-7] in delete_name_list:
                continue
            elif not train and not test and name[7:-7] not in delete_name_list:
                continue
            reverse = '0' if name[5] == '1' else '1'
            self.file_list.append([join(folder_a, f'flair{name[5]}_after', name[7:-7]), join(folder_a, f'flair{name[5]}_after', name[7:-7], 'T1C.nii.gz')])
            self.label_list.append(1)
            self.file_list.append([join(folder_a, f'flair{name[5]}_after', name[7:-7]), join(folder_b, f'{name[:-7]}.nii.gz')])
            self.label_list.append(1)
            self.file_list.append([join(folder_a, f'flair{name[5]}_after', name[7:-7]), join(folder_c, f'label{reverse}_{name[7:-7]}.nii.gz')])
            self.label_list.append(0)
        print('label1 : ',sum(self.label_list), 'label0 : ', len(self.label_list)-sum(self.label_list))
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        T1, _, _, _ = NiiDataRead(join(self.file_list[idx][0], 'T1.nii.gz'))
        # T2F, _, _, _ = NiiDataRead(join(self.file_list[idx][0], 'T2F.nii.gz'))
        T2F, _, _, _ = NiiDataRead(join(self.file_list[idx][0], 'T1C.nii.gz'))
        synthsis, _, _, _ = NiiDataRead(self.file_list[idx][1])
        label = self.label_list[idx]

        T1 = harmonize_mr(T1)
        T2F = harmonize_mr(T2F)
        synthsis = harmonize_mr(synthsis)

        if T1.shape[0]-12 <= 0:
            z_start = [0]
        else:
            z_start = np.random.randint(0, T1.shape[0]-12, 1)
        T1 = T1[z_start[0]:z_start[0]+12, :, :]
        T2F = T2F[z_start[0]:z_start[0]+12, :, :]
        synthsis = synthsis[z_start[0]:z_start[0]+12, :, :]

        assert T1.shape[0] == 12
        assert T2F.shape[0] == 12
        assert synthsis.shape[0] == 12
        Real = torch.from_numpy(np.concatenate((T1[np.newaxis, ...], T2F[np.newaxis, ...]), 0)).float()
        synthsis = torch.from_numpy(synthsis)

        """if self.aug:
            # random Horizontal and Vertical Flip to both image and mask
            p1 = np.random.choice([0, 1])
            p2 = np.random.choice([0, 1])
            self.trans = transforms.Compose([
                transforms.RandomHorizontalFlip(p1),
                transforms.RandomVerticalFlip(p2),
                transforms.RandomRotation(10, resample=False, expand=False, center=None),
            ])
            Real = self.trans(Real)
            synthsis = self.trans(synthsis)"""
        # Convert data to PyTorch tensor
        # Real = Real.unsqueeze(0).float()
        synthsis = synthsis.unsqueeze(0).float()

        # if self.aug:
        #     # random Horizontal and Vertical Flip to both image and mask
        #     p1 = np.random.choice([0, 1])
        #     p2 = np.random.choice([0, 1])
        #     self.trans = transforms.Compose([
        #         transforms.RandomHorizontalFlip(p1),
        #         transforms.RandomVerticalFlip(p2),
        #         transforms.RandomRotation(10, resample=False, expand=False, center=None),
        #     ])
        #     Real = self.trans(Real)
        #     synthsis = self.trans(synthsis)
        # # Convert data to PyTorch tensor
        # Real = torch.from_numpy(Real).unsqueeze(0).float()
        # synthsis = torch.from_numpy(synthsis).unsqueeze(0).float()

        data = torch.cat((Real, synthsis), dim=0)
        label = torch.tensor(label).unsqueeze(0).float()
        return data, label



class ThreeFolderDataset_t1conly(Dataset):
    def __init__(self, folder_a, folder_b, folder_c, train, cv=None, test=False, fold=0):
        self.folder_a = folder_a
        self.folder_b = folder_b
        self.folder_c = folder_c
        self.files_b = sorted(os.listdir(folder_b))
        # assert len(self.files_b) == len(self.files_b) == len(self.files_c), "All folders must have same number of files"

        if not test:
            # 读取txt文件
            with open(cv, 'r') as f:
                delete_name_list = f.readlines()
                delete_name_list = [i.strip('\n') for i in delete_name_list]
                delete_name_list = [i.split('\t')[fold] for i in delete_name_list]

        self.file_list = []
        self.label_list = []
        for name in self.files_b:
            if train and name[7:-7] in delete_name_list:
                continue
            elif not train and not test and name[7:-7] not in delete_name_list:
                continue
            reverse = '0' if name[5] == '1' else '1'
            self.file_list.append([join(folder_a, f'flair{name[5]}_after', name[7:-7]), join(folder_b, f'{name[:-7]}.nii.gz')])
            self.label_list.append(1)
            self.file_list.append([join(folder_a, f'flair{name[5]}_after', name[7:-7]), join(folder_c, f'label{reverse}_{name[7:-7]}.nii.gz')])
            self.label_list.append(0)
        print('label1 : ',sum(self.label_list), 'label0 : ', len(self.label_list)-sum(self.label_list))
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # T2F, _, _, _ = NiiDataRead(join(self.file_list[idx][0], 'T2F.nii.gz'))
        T2F, _, _, _ = NiiDataRead(join(self.file_list[idx][0], 'T1C.nii.gz'))
        synthsis, _, _, _ = NiiDataRead(self.file_list[idx][1])
        label = self.label_list[idx]


        T2F = harmonize_mr(T2F)
        synthsis = harmonize_mr(synthsis)

        if T2F.shape[0]-12 <= 0:
            z_start = [0]
        else:
            z_start = np.random.randint(0, T2F.shape[0]-12, 1)
        T2F = T2F[z_start[0]:z_start[0]+12, :, :]
        synthsis = synthsis[z_start[0]:z_start[0]+12, :, :]


        assert T2F.shape[0] == 12
        assert synthsis.shape[0] == 12

        T2F = torch.from_numpy(T2F)
        synthsis = torch.from_numpy(synthsis)

        """if self.aug:
            # random Horizontal and Vertical Flip to both image and mask
            p1 = np.random.choice([0, 1])
            p2 = np.random.choice([0, 1])
            self.trans = transforms.Compose([
                transforms.RandomHorizontalFlip(p1),
                transforms.RandomVerticalFlip(p2),
                transforms.RandomRotation(10, resample=False, expand=False, center=None),
            ])
            Real = self.trans(Real)
            synthsis = self.trans(synthsis)"""
        # Convert data to PyTorch tensor
        T2F = T2F.unsqueeze(0).float()
        synthsis = synthsis.unsqueeze(0).float()

        # if self.aug:
        #     # random Horizontal and Vertical Flip to both image and mask
        #     p1 = np.random.choice([0, 1])
        #     p2 = np.random.choice([0, 1])
        #     self.trans = transforms.Compose([
        #         transforms.RandomHorizontalFlip(p1),
        #         transforms.RandomVerticalFlip(p2),
        #         transforms.RandomRotation(10, resample=False, expand=False, center=None),
        #     ])
        #     Real = self.trans(Real)
        #     synthsis = self.trans(synthsis)
        # # Convert data to PyTorch tensor
        # Real = torch.from_numpy(Real).unsqueeze(0).float()
        # synthsis = torch.from_numpy(synthsis).unsqueeze(0).float()

        data = torch.cat((T2F, synthsis), dim=0)
        label = torch.tensor(label).unsqueeze(0).float()
        return data, label



class ThreeFolderDataset_2D_t1c(Dataset):
    def __init__(self, folder_a, folder_b, folder_c, train, cv=None, test=False, fold=0):
        self.folder_a = folder_a
        self.aug = train
        self.aug = False
        files_a_0 = os.listdir(join(folder_a, 'flair0_after'))
        files_a_1 = os.listdir(join(folder_a, 'flair1_after'))
        # files_a_0 = []
        # files_a_1 = []
        self.files_a = files_a_0 + files_a_1
        self.labels = [0]*len(files_a_0) + [1]*len(files_a_1)
        self.reals = []
        self.synthsis = []
        self.label_list = []
        if not test:
            # 读取txt文件
            with open(cv, 'r') as f:
                delete_name_list = f.readlines()
                delete_name_list = [i.strip('\n') for i in delete_name_list]
                delete_name_list = [i.split('\t')[fold] for i in delete_name_list]
        count = 0
        for i, name in enumerate(self.files_a):
            if train and name in delete_name_list:
                continue
            elif not train and not test and name not in delete_name_list:
                continue
            # count += 1
            # if count >=10:
            #     break
            label = self.labels[i]
            reverse = 0 if label == 1 else 1
            Real, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/T1C.nii.gz'))
            Fake, _, _, _ = NiiDataRead(join(folder_b, f'label{label}_{name}.nii.gz'))
            Fake_reverse, _, _, _ = NiiDataRead(join(folder_c, f'label{reverse}_{name}.nii.gz'))
            ROI, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/ROI.nii.gz'))

            for slice in np.unique(ROI.nonzero()[0]):
                self.reals.append(Real[slice])
                self.synthsis.append(Fake[slice])
                self.label_list.append(1)

                self.reals.append(Real[slice])
                self.synthsis.append(Fake_reverse[slice])
                self.label_list.append(0)
        print('label1 : ', sum(self.label_list), 'label0 : ', len(self.label_list) - sum(self.label_list))

    def __len__(self):
        return len(self.reals)

    def __getitem__(self, idx):
        Real = self.reals[idx]
        synthsis = self.synthsis[idx]
        label = self.label_list[idx]

        Real = harmonize_mr(Real)
        synthsis = harmonize_mr(synthsis)
        Real = torch.from_numpy(Real).unsqueeze(0).float()
        synthsis = torch.from_numpy(synthsis).unsqueeze(0).float()
        if self.aug:
            # random Horizontal and Vertical Flip to both image and mask
            p1 = np.random.choice([0, 1])
            p2 = np.random.choice([0, 1])
            self.trans = transforms.Compose([
                transforms.RandomHorizontalFlip(p1),
                transforms.RandomVerticalFlip(p2),
                # transforms.RandomRotation(10, resample=False, expand=False, center=None),
                transforms.RandomRotation(10, expand=False, center=None),
            ])
            Real = self.trans(Real)
            synthsis = self.trans(synthsis)
        # Convert data to PyTorch tensor

        data = torch.cat((Real, synthsis), dim=0)
        label = torch.tensor(label).unsqueeze(0).float()
        return data, label


class ThreeFolderDataset_2D_t1_t1c(Dataset):
    def __init__(self, folder_a, folder_b, folder_c, train, cv=None, test=False, fold=0):
        self.folder_a = folder_a
        self.aug = train
        files_a_0 = os.listdir(join(folder_a, 'flair0_after'))
        files_a_1 = os.listdir(join(folder_a, 'flair1_after'))
        # files_a_0 = []
        # files_a_1 = []
        self.files_a = files_a_0 + files_a_1
        self.labels = [0]*len(files_a_0) + [1]*len(files_a_1)
        self.reals_t1 = []
        self.reals_t1c = []
        self.synthsis = []
        self.label_list = []
        if not test:
            # 读取txt文件
            with open(cv, 'r') as f:
                delete_name_list = f.readlines()
                delete_name_list = [i.strip('\n') for i in delete_name_list]
                delete_name_list = [i.split('\t')[fold] for i in delete_name_list]
        count = 0
        for i, name in enumerate(self.files_a):
            if train and name in delete_name_list:
                continue
            elif not train and not test and name not in delete_name_list:
                continue
            # count += 1
            # if count >=2:
            #     break
            label = self.labels[i]
            reverse = 0 if label == 1 else 1
            T1, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/T1.nii.gz'))
            T1C, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/T1C.nii.gz'))
            Fake, _, _, _ = NiiDataRead(join(folder_b, f'label{label}_{name}.nii.gz'))
            Fake_reverse, _, _, _ = NiiDataRead(join(folder_c, f'label{reverse}_{name}.nii.gz'))
            ROI, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/ROI.nii.gz'))

            T1 = harmonize_mr(T1)
            T1C = harmonize_mr(T1C)
            Fake = harmonize_mr(Fake)
            Fake_reverse = harmonize_mr(Fake_reverse)

            for slice in np.unique(ROI.nonzero()[0]):
                self.reals_t1.append(T1[slice])
                self.reals_t1c.append(T1C[slice])
                self.synthsis.append(Fake[slice])
                self.label_list.append(1)

                self.reals_t1.append(T1[slice])
                self.reals_t1c.append(T1C[slice])
                self.synthsis.append(Fake_reverse[slice])
                self.label_list.append(0)
        print('label1 : ', sum(self.label_list), 'label0 : ', len(self.label_list) - sum(self.label_list))

    def __len__(self):
        return len(self.reals_t1)

    def __getitem__(self, idx):
        T1 = self.reals_t1[idx]
        T1C = self.reals_t1c[idx]
        synthsis = self.synthsis[idx]
        label = self.label_list[idx]

        # Real = harmonize_mr(Real)
        # synthsis = harmonize_mr(synthsis)
        T1 = torch.from_numpy(T1).unsqueeze(0).float()
        T1C = torch.from_numpy(T1C).unsqueeze(0).float()
        synthsis = torch.from_numpy(synthsis).unsqueeze(0).float()
        if self.aug:
            # random Horizontal and Vertical Flip to both image and mask
            p1 = np.random.choice([0, 1])
            p2 = np.random.choice([0, 1])
            self.trans = transforms.Compose([
                transforms.RandomHorizontalFlip(p1),
                transforms.RandomVerticalFlip(p2),
                # transforms.RandomRotation(10, resample=False, expand=False, center=None),
                transforms.RandomRotation(10, expand=False, center=None),
            ])
            T1 = self.trans(T1)
            T1C = self.trans(T1C)
            synthsis = self.trans(synthsis)
        # Convert data to PyTorch tensor

        data = torch.cat((T1, T1C, synthsis), dim=0)
        label = torch.tensor(label).unsqueeze(0).float()
        return data, label


class ThreeFolderDataset_2D_t1_t2f_t1c(Dataset):
    def __init__(self, folder_a, folder_b, folder_c, train, cv=None, test=False, fold=0):
        self.folder_a = folder_a
        self.aug = train
        files_a_0 = os.listdir(join(folder_a, 'flair0_after'))
        files_a_1 = os.listdir(join(folder_a, 'flair1_after'))
        # files_a_0 = []
        # files_a_1 = []
        self.files_a = files_a_0 + files_a_1
        self.labels = [0]*len(files_a_0) + [1]*len(files_a_1)
        self.reals_t1 = []
        self.reals_t2f = []
        self.reals_t1c = []
        self.synthsis = []
        self.label_list = []
        if not test:
            # 读取txt文件
            with open(cv, 'r') as f:
                delete_name_list = f.readlines()
                delete_name_list = [i.strip('\n') for i in delete_name_list]
                delete_name_list = [i.split('\t')[fold] for i in delete_name_list]
        count = 0
        for i, name in enumerate(self.files_a):
            if train and name in delete_name_list:
                continue
            elif not train and not test and name not in delete_name_list:
                continue
            # count += 1
            # if count >=10:
            #     break
            label = self.labels[i]
            reverse = 0 if label == 1 else 1
            T1, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/T1.nii.gz'))
            T2F, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/T2F.nii.gz'))
            T1C, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/T1C.nii.gz'))
            Fake, _, _, _ = NiiDataRead(join(folder_b, f'label{label}_{name}.nii.gz'))
            Fake_reverse, _, _, _ = NiiDataRead(join(folder_c, f'label{reverse}_{name}.nii.gz'))
            ROI, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/ROI.nii.gz'))

            T1 = harmonize_mr(T1)
            T2F = harmonize_mr(T2F)
            T1C = harmonize_mr(T1C)
            Fake = harmonize_mr(Fake)
            Fake_reverse = harmonize_mr(Fake_reverse)

            for slice in np.unique(ROI.nonzero()[0]):
                self.reals_t1.append(T1[slice])
                self.reals_t2f.append(T2F[slice])
                self.reals_t1c.append(T1C[slice])
                self.synthsis.append(Fake[slice])
                self.label_list.append(1)

                self.reals_t1.append(T1[slice])
                self.reals_t2f.append(T2F[slice])
                self.reals_t1c.append(T1C[slice])
                self.synthsis.append(Fake_reverse[slice])
                self.label_list.append(0)
        print('label1 : ', sum(self.label_list), 'label0 : ', len(self.label_list) - sum(self.label_list))

    def __len__(self):
        return len(self.reals_t1)

    def __getitem__(self, idx):
        T1 = self.reals_t1[idx]
        T2F = self.reals_t2f[idx]
        T1C = self.reals_t1c[idx]
        synthsis = self.synthsis[idx]
        label = self.label_list[idx]

        # Real = harmonize_mr(Real)
        # synthsis = harmonize_mr(synthsis)
        T1 = torch.from_numpy(T1).unsqueeze(0).float()
        T2F = torch.from_numpy(T2F).unsqueeze(0).float()
        T1C = torch.from_numpy(T1C).unsqueeze(0).float()
        synthsis = torch.from_numpy(synthsis).unsqueeze(0).float()
        if self.aug:
            # random Horizontal and Vertical Flip to both image and mask
            p1 = np.random.choice([0, 1])
            p2 = np.random.choice([0, 1])
            self.trans = transforms.Compose([
                transforms.RandomHorizontalFlip(p1),
                transforms.RandomVerticalFlip(p2),
                # transforms.RandomRotation(10, resample=False, expand=False, center=None),
                transforms.RandomRotation(10, expand=False, center=None),
            ])
            T1 = self.trans(T1)
            T2F = self.trans(T2F)
            T1C = self.trans(T1C)
            synthsis = self.trans(synthsis)
        # Convert data to PyTorch tensor

        data = torch.cat((T1, T2F, T1C, synthsis), dim=0)
        label = torch.tensor(label).unsqueeze(0).float()
        return data, label



class ThreeFolderDataset_2D_t1_t2f_t1c2_ERTguided(Dataset):
    def __init__(self, folder_a, folder_b, folder_c, train, cv=None, test=False, fold=0):
        self.folder_a = folder_a
        self.aug = train
        files_a_0 = os.listdir(join(folder_a, 'flair0_after'))
        files_a_1 = os.listdir(join(folder_a, 'flair1_after'))
        # files_a_0 = []
        # files_a_1 = []
        self.files_a = files_a_0 + files_a_1
        self.labels = [0]*len(files_a_0) + [1]*len(files_a_1)
        self.reals_t1 = []
        self.reals_t2f = []
        # self.reals_t1c = []
        self.synthsis_label1 = []
        self.synthsis_label0 = []
        self.label_list = []
        if not test:
            # 读取txt文件
            with open(cv, 'r') as f:
                delete_name_list = f.readlines()
                delete_name_list = [i.strip('\n') for i in delete_name_list]
                delete_name_list = [i.split('\t')[fold] for i in delete_name_list]
        count = 0
        for i, name in enumerate(self.files_a):
            if train and name in delete_name_list:
                continue
            elif not train and not test and name not in delete_name_list:
                continue
            # count += 1
            # if count >=64:
            #     break
            label = self.labels[i]
            reverse = 0 if label == 1 else 1
            T1, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/T1.nii.gz'))
            T2F, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/T2F.nii.gz'))
            # T1C, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/T1C.nii.gz'))
            Fake, _, _, _ = NiiDataRead(join(folder_b, f'label{label}_{name}.nii.gz'))
            Fake_reverse, _, _, _ = NiiDataRead(join(folder_c, f'label{reverse}_{name}.nii.gz'))
            ROI, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/ROI.nii.gz'))

            T1 = harmonize_mr(T1)
            T2F = harmonize_mr(T2F)
            # T1C = harmonize_mr(T1C)
            Fake = harmonize_mr(Fake)
            Fake_reverse = harmonize_mr(Fake_reverse)

            for slice in np.unique(ROI.nonzero()[0]):
                self.reals_t1.append(T1[slice])
                self.reals_t2f.append(T2F[slice])
                # self.reals_t1c.append(T1C[slice])
                if label == 1:
                    self.synthsis_label1.append(Fake[slice])
                    self.synthsis_label0.append(Fake_reverse[slice])
                elif label == 0:
                    self.synthsis_label1.append(Fake_reverse[slice])
                    self.synthsis_label0.append(Fake[slice])
                self.label_list.append(label)
        print('label1 : ', sum(self.label_list), 'label0 : ', len(self.label_list) - sum(self.label_list))

    def __len__(self):
        return len(self.reals_t1)

    def __getitem__(self, idx):
        T1 = self.reals_t1[idx]
        T2F = self.reals_t2f[idx]
        # T1C = self.reals_t1c[idx]
        synthsis_label1 = self.synthsis_label1[idx]
        synthsis_label0 = self.synthsis_label0[idx]
        label = self.label_list[idx]

        # Real = harmonize_mr(Real)
        # synthsis = harmonize_mr(synthsis)
        T1 = torch.from_numpy(T1).unsqueeze(0).float()
        T2F = torch.from_numpy(T2F).unsqueeze(0).float()
        # T1C = torch.from_numpy(T1C).unsqueeze(0).float()
        synthsis_label1 = torch.from_numpy(synthsis_label1).unsqueeze(0).float()
        synthsis_label0 = torch.from_numpy(synthsis_label0).unsqueeze(0).float()
        if self.aug:
            # random Horizontal and Vertical Flip to both image and mask
            p1 = np.random.choice([0, 1])
            p2 = np.random.choice([0, 1])
            self.trans = transforms.Compose([
                transforms.RandomHorizontalFlip(p1),
                transforms.RandomVerticalFlip(p2),
                # transforms.RandomRotation(10, resample=False, expand=False, center=None),
                transforms.RandomRotation(10, expand=False, center=None),
            ])
            T1 = self.trans(T1)
            T2F = self.trans(T2F)
            # T1C = self.trans(T1C)
            synthsis_label1 = self.trans(synthsis_label1)
            synthsis_label0 = self.trans(synthsis_label0)

        # Convert data to PyTorch tensor

        data = torch.cat((T1, T2F, synthsis_label1, synthsis_label0), dim=0)
        label = torch.tensor(label).unsqueeze(0).float()
        return data, label



class ThreeFolderDataset_2D_t1_t2f_t1c_plus(Dataset):
    def __init__(self, folder_a, folder_b, folder_c, train, cv=None, test=False, fold=0):
        self.folder_a = folder_a
        self.aug = train
        files_a_0 = os.listdir(join(folder_a, 'flair0_after'))
        files_a_1 = os.listdir(join(folder_a, 'flair1_after'))
        # files_a_0 = []
        # files_a_1 = []
        self.files_a = files_a_0 + files_a_1
        self.labels = [0]*len(files_a_0) + [1]*len(files_a_1)
        self.reals_t1 = []
        self.reals_t2f = []
        self.reals_t1c = []
        self.synthsis = []
        self.label_list = []
        if not test:
            # 读取txt文件
            with open(cv, 'r') as f:
                delete_name_list = f.readlines()
                delete_name_list = [i.strip('\n') for i in delete_name_list]
                delete_name_list = [i.split('\t')[fold] for i in delete_name_list]
        count = 0
        for i, name in enumerate(self.files_a):
            if train and name in delete_name_list:
                continue
            elif not train and not test and name not in delete_name_list:
                continue
            # count += 1
            # if count >=10:
            #     break
            label = self.labels[i]
            reverse = 0 if label == 1 else 1
            T1, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/T1.nii.gz'))
            T2F, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/T2F.nii.gz'))
            T1C, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/T1C.nii.gz'))
            Fake, _, _, _ = NiiDataRead(join(folder_b, f'label{label}_{name}.nii.gz'))
            Fake_reverse, _, _, _ = NiiDataRead(join(folder_c, f'label{reverse}_{name}.nii.gz'))
            ROI, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/ROI.nii.gz'))

            T1 = harmonize_mr(T1)
            T2F = harmonize_mr(T2F)
            T1C = harmonize_mr(T1C)
            Fake = harmonize_mr(Fake)
            Fake_reverse = harmonize_mr(Fake_reverse)

            for slice in np.unique(ROI.nonzero()[0]):
                self.reals_t1.append(T1[slice])
                self.reals_t2f.append(T2F[slice])
                self.reals_t1c.append(T1C[slice])
                self.synthsis.append(Fake[slice])
                self.label_list.append(1)

                self.reals_t1.append(T1[slice])
                self.reals_t2f.append(T2F[slice])
                self.reals_t1c.append(T1C[slice])
                self.synthsis.append(T1C[slice])
                self.label_list.append(1)

                self.reals_t1.append(T1[slice])
                self.reals_t2f.append(T2F[slice])
                self.reals_t1c.append(T1C[slice])
                self.synthsis.append(Fake_reverse[slice])
                self.label_list.append(0)
        print('label1 : ', sum(self.label_list), 'label0 : ', len(self.label_list) - sum(self.label_list))

    def __len__(self):
        return len(self.reals_t1)

    def __getitem__(self, idx):
        T1 = self.reals_t1[idx]
        T2F = self.reals_t2f[idx]
        T1C = self.reals_t1c[idx]
        synthsis = self.synthsis[idx]
        label = self.label_list[idx]

        # Real = harmonize_mr(Real)
        # synthsis = harmonize_mr(synthsis)
        T1 = torch.from_numpy(T1).unsqueeze(0).float()
        T2F = torch.from_numpy(T2F).unsqueeze(0).float()
        T1C = torch.from_numpy(T1C).unsqueeze(0).float()
        synthsis = torch.from_numpy(synthsis).unsqueeze(0).float()
        if self.aug:
            # random Horizontal and Vertical Flip to both image and mask
            p1 = np.random.choice([0, 1])
            p2 = np.random.choice([0, 1])
            self.trans = transforms.Compose([
                transforms.RandomHorizontalFlip(p1),
                transforms.RandomVerticalFlip(p2),
                # transforms.RandomRotation(10, resample=False, expand=False, center=None),
                transforms.RandomRotation(10, expand=False, center=None),
            ])
            T1 = self.trans(T1)
            T2F = self.trans(T2F)
            T1C = self.trans(T1C)
            synthsis = self.trans(synthsis)
        # Convert data to PyTorch tensor

        data = torch.cat((T1, T2F, T1C, synthsis), dim=0)
        label = torch.tensor(label).unsqueeze(0).float()
        return data, label



class ThreeFolderDataset_2D_new(Dataset):
    def __init__(self, folder_a, folder_b, folder_c, train, cv=None, test=False, fold=0):
        self.folder_a = folder_a
        self.aug = train
        files_a_0 = os.listdir(join(folder_a, 'flair0_after'))
        files_a_1 = os.listdir(join(folder_a, 'flair1_after'))
        # files_a_0 = []
        # files_a_1 = []
        self.files_a = files_a_0 + files_a_1
        self.labels = [0]*len(files_a_0) + [1]*len(files_a_1)
        self.reals_t1 = []
        self.reals_t2f = []
        self.synthsis = []
        self.label_list = []
        if not test:
            # 读取txt文件
            with open(cv, 'r') as f:
                delete_name_list = f.readlines()
                delete_name_list = [i.strip('\n') for i in delete_name_list]
                delete_name_list = [i.split('\t')[fold] for i in delete_name_list]
        count = 0
        for i, name in enumerate(self.files_a):
            if train and name in delete_name_list:
                continue
            elif not train and not test and name not in delete_name_list:
                continue
            # count += 1
            # if count >=10:
            #     break
            label = self.labels[i]
            reverse = 0 if label == 1 else 1
            T1, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/T1.nii.gz'))
            T2F, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/T2F.nii.gz'))
            Fake, _, _, _ = NiiDataRead(join(folder_b, f'label{label}_{name}.nii.gz'))
            Fake_reverse, _, _, _ = NiiDataRead(join(folder_c, f'label{reverse}_{name}.nii.gz'))
            ROI, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/ROI.nii.gz'))

            T1 = harmonize_mr(T1)
            T2F = harmonize_mr(T2F)
            Fake = harmonize_mr(Fake)
            Fake_reverse = harmonize_mr(Fake_reverse)

            for slice in np.unique(ROI.nonzero()[0]):
                self.reals_t1.append(T1[slice])
                self.reals_t2f.append(T2F[slice])
                self.synthsis.append(Fake[slice])
                self.label_list.append(1)

                self.reals_t1.append(T1[slice])
                self.reals_t2f.append(T2F[slice])
                self.synthsis.append(Fake_reverse[slice])
                self.label_list.append(0)
        print('label1 : ', sum(self.label_list), 'label0 : ', len(self.label_list) - sum(self.label_list))

    def __len__(self):
        return len(self.reals_t1)

    def __getitem__(self, idx):
        T1 = self.reals_t1[idx]
        T2F = self.reals_t2f[idx]
        synthsis = self.synthsis[idx]
        label = self.label_list[idx]

        # Real = harmonize_mr(Real)
        # synthsis = harmonize_mr(synthsis)
        T1 = torch.from_numpy(T1).unsqueeze(0).float()
        T2F = torch.from_numpy(T2F).unsqueeze(0).float()
        synthsis = torch.from_numpy(synthsis).unsqueeze(0).float()
        if self.aug:
            # random Horizontal and Vertical Flip to both image and mask
            p1 = np.random.choice([0, 1])
            p2 = np.random.choice([0, 1])
            self.trans = transforms.Compose([
                transforms.RandomHorizontalFlip(p1),
                transforms.RandomVerticalFlip(p2),
                # transforms.RandomRotation(10, resample=False, expand=False, center=None),
                transforms.RandomRotation(10, expand=False, center=None),
            ])
            T1 = self.trans(T1)
            T2F = self.trans(T2F)
            synthsis = self.trans(synthsis)
        # Convert data to PyTorch tensor

        data = torch.cat((T1, T2F, synthsis), dim=0)
        label = torch.tensor(label).unsqueeze(0).float()
        return data, label


class ThreeFolderDataset_2D_new_plus(Dataset):
    def __init__(self, folder_a, folder_b, folder_c, train, cv=None, test=False, fold=0):
        self.folder_a = folder_a
        self.aug = train
        files_a_0 = os.listdir(join(folder_a, 'flair0_after'))
        files_a_1 = os.listdir(join(folder_a, 'flair1_after'))
        # files_a_0 = []
        # files_a_1 = []
        self.files_a = files_a_0 + files_a_1
        self.labels = [0]*len(files_a_0) + [1]*len(files_a_1)
        self.reals_t1 = []
        self.reals_t2f = []
        self.synthsis = []
        self.label_list = []
        if not test:
            # 读取txt文件
            with open(cv, 'r') as f:
                delete_name_list = f.readlines()
                delete_name_list = [i.strip('\n') for i in delete_name_list]
                delete_name_list = [i.split('\t')[fold] for i in delete_name_list]
        count = 0
        for i, name in enumerate(self.files_a):
            if train and name in delete_name_list:
                continue
            elif not train and not test and name not in delete_name_list:
                continue
            # count += 1
            # if count >=2:
            #     break
            label = self.labels[i]
            reverse = 0 if label == 1 else 1
            T1, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/T1.nii.gz'))
            T1C, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/T1C.nii.gz'))
            T2F, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/T2F.nii.gz'))
            Fake, _, _, _ = NiiDataRead(join(folder_b, f'label{label}_{name}.nii.gz'))
            Fake_reverse, _, _, _ = NiiDataRead(join(folder_c, f'label{reverse}_{name}.nii.gz'))
            ROI, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/ROI.nii.gz'))

            T1 = harmonize_mr(T1)
            T1C = harmonize_mr(T1C)
            T2F = harmonize_mr(T2F)
            Fake = harmonize_mr(Fake)
            Fake_reverse = harmonize_mr(Fake_reverse)

            for slice in np.unique(ROI.nonzero()[0]):
                self.reals_t1.append(T1[slice])
                self.reals_t2f.append(T2F[slice])
                self.synthsis.append(Fake[slice])
                self.label_list.append(1)

                self.reals_t1.append(T1[slice])
                self.reals_t2f.append(T2F[slice])
                self.synthsis.append(T1C[slice])
                self.label_list.append(1)

                self.reals_t1.append(T1[slice])
                self.reals_t2f.append(T2F[slice])
                self.synthsis.append(Fake_reverse[slice])
                self.label_list.append(0)
        print('label1 : ', sum(self.label_list), 'label0 : ', len(self.label_list) - sum(self.label_list))

    def __len__(self):
        return len(self.reals_t1)

    def __getitem__(self, idx):
        T1 = self.reals_t1[idx]
        T2F = self.reals_t2f[idx]
        synthsis = self.synthsis[idx]
        label = self.label_list[idx]

        # Real = harmonize_mr(Real)
        # synthsis = harmonize_mr(synthsis)
        T1 = torch.from_numpy(T1).unsqueeze(0).float()
        T2F = torch.from_numpy(T2F).unsqueeze(0).float()
        synthsis = torch.from_numpy(synthsis).unsqueeze(0).float()
        if self.aug:
            # random Horizontal and Vertical Flip to both image and mask
            p1 = np.random.choice([0, 1])
            p2 = np.random.choice([0, 1])
            self.trans = transforms.Compose([
                transforms.RandomHorizontalFlip(p1),
                transforms.RandomVerticalFlip(p2),
                # transforms.RandomRotation(10, resample=False, expand=False, center=None),
                transforms.RandomRotation(10, expand=False, center=None),
            ])
            T1 = self.trans(T1)
            T2F = self.trans(T2F)
            synthsis = self.trans(synthsis)
        # Convert data to PyTorch tensor

        data = torch.cat((T1, T2F, synthsis), dim=0)
        label = torch.tensor(label).unsqueeze(0).float()
        return data, label



def test_pred_2D(net, Real, Fake, Fake_reverse, ROI, opt):
    pred_hope1 = torch.zeros(1).cuda()
    pred_list_hope1 = torch.zeros(1).cuda()
    pred_hope0 = torch.zeros(1).cuda()
    pred_list_hope0 = torch.zeros(1).cuda()
    """
    pred_hope1 = torch.zeros(1)
    pred_list_hope1 = torch.zeros(1)
    pred_hope0 = torch.zeros(1)
    pred_list_hope0 = torch.zeros(1)"""
    if isinstance(Real, dict):
        for image_name in Real.keys():
            Real[image_name] = harmonize_mr(Real[image_name])
    else:
        Real = harmonize_mr(Real)
    Fake = harmonize_mr(Fake)
    Fake_reverse = harmonize_mr(Fake_reverse)
    image_list_hope1 = []
    image_list_hope0 = []
    count = 0
    for slice in np.unique(ROI.nonzero()[0]):
        if isinstance(Real, dict):
            try:
                data_hope1 = np.stack((Real['t1'][slice], Real['t2f'][slice], Real['t1c'][slice], Fake[slice]), axis=0)
                data_hope0 = np.stack((Real['t1'][slice], Real['t2f'][slice], Real['t1c'][slice], Fake_reverse[slice]), axis=0)
            except:
                data_hope1 = np.stack((Real['t1'][slice], Real['t2f'][slice], Fake[slice]), axis=0)
                data_hope0 = np.stack((Real['t1'][slice], Real['t2f'][slice], Fake_reverse[slice]), axis=0)
                # data_hope1 = np.stack((Real['t1'][slice], Real['t1c'][slice], Fake[slice]), axis=0)
                # data_hope0 = np.stack((Real['t1'][slice], Real['t1c'][slice], Fake_reverse[slice]), axis=0)
        else:
            data_hope1 = np.stack((Real[slice], Fake[slice]), axis=0)
            data_hope0 = np.stack((Real[slice], Fake_reverse[slice]), axis=0)
        image_list_hope1.append(data_hope1)
        image_list_hope0.append(data_hope0)
        count += 1
    image_hope1 = np.array(image_list_hope1)
    image_hope0 = np.array(image_list_hope0)

    n_num = image_hope1.shape[0] // opt.batch_size
    n_num = n_num + 0 if image_hope1.shape[0] % opt.batch_size == 0 else n_num + 1

    for n in range(n_num):
        # print(f'{n + 1}/{n_num}', end=' || ')
        if n == n_num - 1:
            one_image = image_hope1[n * opt.batch_size:, :, :, :]
        else:
            one_image = image_hope1[n * opt.batch_size: (n + 1) * opt.batch_size, :, :, :]
        one_image = torch.from_numpy(one_image).float().cuda()
        """one_image = torch.from_numpy(one_image).float()"""
        y = torch.sigmoid(net(one_image))
        k = torch.zeros_like(y)
        k[y >= 0.5] = 1
        pred_hope1 += torch.sum(y.detach(), dim=0)
        pred_list_hope1 += torch.sum(k.detach(), dim=0)

    for n in range(n_num):
        print(f'{n + 1}/{n_num}', end=' || ')
        if n == n_num - 1:
            one_image = image_hope0[n * opt.batch_size:, :, :, :]
        else:
            one_image = image_hope0[n * opt.batch_size: (n + 1) * opt.batch_size, :, :, :]
        one_image = torch.from_numpy(one_image).float().cuda()
        """one_image = torch.from_numpy(one_image).float()"""

        y = torch.sigmoid(net(one_image))
        k = torch.zeros_like(y)
        k[y >= 0.5] = 1
        pred_hope0 += torch.sum(y.detach(), dim=0)
        pred_list_hope0 += torch.sum(k.detach(), dim=0)

    label1 = torch.tensor(int(1)).float().unsqueeze(0).cuda()
    pred_list1 = pred_list_hope1 / count
    label0 = torch.tensor(int(0)).float().unsqueeze(0).cuda()
    pred_list0 = pred_list_hope0 / count
    return (pred_list1, pred_list0), (label1, label0)



def test_pred_2D_ERTguided(net, Real, Fake, Fake_reverse, ROI, opt, label):
    """
    pred_hope1 = torch.zeros(1)
    pred_list_hope1 = torch.zeros(1)
    pred_hope0 = torch.zeros(1)
    pred_list_hope0 = torch.zeros(1)"""
    pred = torch.zeros(1).cuda()
    pred_list = torch.zeros(1).cuda()
    if isinstance(Real, dict):
        for image_name in Real.keys():
            Real[image_name] = harmonize_mr(Real[image_name])
    else:
        Real = harmonize_mr(Real)
    Fake = harmonize_mr(Fake)
    Fake_reverse = harmonize_mr(Fake_reverse)
    image_list = []
    count = 0
    for slice in np.unique(ROI.nonzero()[0]):
        if isinstance(Real, dict):
            if label==1:
                data = np.stack((Real['t1'][slice], Real['t2f'][slice], Fake[slice], Fake_reverse[slice]), axis=0)
            elif label==0:
                data = np.stack((Real['t1'][slice], Real['t2f'][slice], Fake_reverse[slice], Fake[slice]), axis=0)
        count += 1
        image_list.append(data)
    image = np.array(image_list)
    n_num = image.shape[0] // opt.batch_size
    n_num = n_num + 0 if image.shape[0] % opt.batch_size == 0 else n_num + 1

    for n in range(n_num):
        # print(f'{n + 1}/{n_num}', end=' || ')
        if n == n_num - 1:
            one_image = image[n * opt.batch_size:, :, :, :]
        else:
            one_image = image[n * opt.batch_size: (n + 1) * opt.batch_size, :, :, :]
        one_image = torch.from_numpy(one_image).float().cuda()
        """one_image = torch.from_numpy(one_image).float()"""
        y = torch.sigmoid(net(one_image))
        k = torch.zeros_like(y)
        k[y >= 0.5] = 1
        pred += torch.sum(y.detach(), dim=0)
        pred_list += torch.sum(k.detach(), dim=0)
    label = torch.tensor(label).float().unsqueeze(0).cuda()
    pred_list = pred_list / count
    return pred_list, label

def test_pred_2D_new(net, Real_t1, Real_t2f, Fake, Fake_reverse, ROI, opt):
    pred_hope1 = torch.zeros(1).cuda()
    pred_list_hope1 = torch.zeros(1).cuda()
    pred_hope0 = torch.zeros(1).cuda()
    pred_list_hope0 = torch.zeros(1).cuda()
    """
    pred_hope1 = torch.zeros(1)
    pred_list_hope1 = torch.zeros(1)
    pred_hope0 = torch.zeros(1)
    pred_list_hope0 = torch.zeros(1)"""

    Real_t1 = harmonize_mr(Real_t1)
    Real_t2f = harmonize_mr(Real_t2f)
    Fake = harmonize_mr(Fake)
    Fake_reverse = harmonize_mr(Fake_reverse)
    image_list_hope1 = []
    image_list_hope0 = []
    count = 0
    for slice in np.unique(ROI.nonzero()[0]):
        data_hope1 = np.stack((Real_t1[slice], Real_t2f[slice], Fake[slice]), axis=0)
        data_hope0 = np.stack((Real_t1[slice], Real_t2f[slice], Fake_reverse[slice]), axis=0)
        image_list_hope1.append(data_hope1)
        image_list_hope0.append(data_hope0)
        count += 1
    image_hope1 = np.array(image_list_hope1)
    image_hope0 = np.array(image_list_hope0)

    n_num = image_hope1.shape[0] // opt.batch_size
    n_num = n_num + 0 if image_hope1.shape[0] % opt.batch_size == 0 else n_num + 1

    for n in range(n_num):
        # print(f'{n + 1}/{n_num}', end=' || ')
        if n == n_num - 1:
            one_image = image_hope1[n * opt.batch_size:, :, :, :]
        else:
            one_image = image_hope1[n * opt.batch_size: (n + 1) * opt.batch_size, :, :, :]
        one_image = torch.from_numpy(one_image).float().cuda()
        """one_image = torch.from_numpy(one_image).float()"""
        y = torch.sigmoid(net(one_image))
        k = torch.zeros_like(y)
        k[y >= 0.5] = 1
        pred_hope1 += torch.sum(y.detach(), dim=0)
        pred_list_hope1 += torch.sum(k.detach(), dim=0)

    for n in range(n_num):
        # print(f'{n + 1}/{n_num}', end=' || ')
        if n == n_num - 1:
            one_image = image_hope0[n * opt.batch_size:, :, :, :]
        else:
            one_image = image_hope0[n * opt.batch_size: (n + 1) * opt.batch_size, :, :, :]
        one_image = torch.from_numpy(one_image).float().cuda()
        """one_image = torch.from_numpy(one_image).float()"""

        y = torch.sigmoid(net(one_image))
        k = torch.zeros_like(y)
        k[y >= 0.5] = 1
        pred_hope0 += torch.sum(y.detach(), dim=0)
        pred_list_hope0 += torch.sum(k.detach(), dim=0)

    label1 = torch.tensor(int(1)).float().unsqueeze(0).cuda()
    pred_list1 = pred_list_hope1 / count
    label0 = torch.tensor(int(0)).float().unsqueeze(0).cuda()
    pred_list0 = pred_list_hope0 / count
    return (pred_list1, pred_list0), (label1, label0)


def test_pred_2D_noreverse(net, Real, ROI, opt):
    pred_logic = torch.zeros(1).cuda()
    pred_binary = torch.zeros(1).cuda()
    image_list = []
    if isinstance(Real, dict):
        for image_name in Real.keys():
            Real[image_name] = harmonize_mr(Real[image_name])
    else:
        Real = harmonize_mr(Real)
    count = 0
    for slice in np.unique(ROI.nonzero()[0]):
        if isinstance(Real, dict):
            try:
                Image = np.stack((Real['t1'][slice], Real['t2f'][slice], Real['t1c'][slice], Fake[slice]), axis=0)
            except:
                Image = np.stack((Real['t1'][slice], Real['t2f'][slice]), axis=0)
        else:
            Image = Real[slice][np.newaxis, ...]
        image_list.append(Image)
        count += 1
    Image = np.array(image_list)
    n_num = Image.shape[0] // opt.batch_size
    n_num = n_num + 0 if Image.shape[0] % opt.batch_size == 0 else n_num + 1
    for n in range(n_num):
        # print(f'{n + 1}/{n_num}', end=' || ')
        if n == n_num - 1:
            one_image = Image[n * opt.batch_size:, :, :, :]
        else:
            one_image = Image[n * opt.batch_size: (n + 1) * opt.batch_size, :, :, :]
        one_image = torch.from_numpy(one_image).float().cuda()
        """one_image = torch.from_numpy(one_image).float()"""
        y = torch.sigmoid(net(one_image))
        k = torch.zeros_like(y)
        k[y >= 0.5] = 1
        pred_logic += torch.sum(y.detach(), dim=0)
        pred_binary += torch.sum(k.detach(), dim=0)

    pred_logic = pred_logic / count
    pred_binary = pred_binary / count
    return pred_logic, pred_binary


"""class ThreeFolderDataset_2Dnew(Dataset):
    def __init__(self, folder_a, folder_b, folder_c, aug=False):
        self.folder_a = folder_a
        self.aug = aug
        files_a_0 = os.listdir(join(folder_a, 'flair0_after'))
        # files_a_1 = os.listdir(join(folder_a, 'flair1_after'))
        # files_a_0 = []
        files_a_1 = []
        self.files_a = files_a_0 + files_a_1
        self.labels = [0]*len(files_a_0) + [1]*len(files_a_1)
        self.reals = []
        self.synthsis = []
        self.label_list = []
        # count = 0
        for i, name in enumerate(self.files_a):
            # count += 1
            # if count >= 10:
            #     break
            label = self.labels[i]
            reverse = 0 if label == 1 else 1
            Real, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/T1C.nii.gz'))
            Fake, _, _, _ = NiiDataRead(join(folder_b, f'label{label}_{name}.nii.gz'))
            Fake_reverse, _, _, _ = NiiDataRead(join(folder_c, f'label{reverse}_{name}.nii.gz'))
            ROI, _, _, _ = NiiDataRead(join(folder_a, f'flair{label}_after/{name}/ROI.nii.gz'))

            for slice in np.unique(ROI.nonzero()[0]):
                self.reals.append(Real[slice])
                self.synthsis.append(Fake[slice])
                self.label_list.append(1)

                self.reals.append(Real[slice])
                self.synthsis.append(Fake_reverse[slice])
                self.label_list.append(0)

    def __len__(self):
        return len(self.reals)

    def __getitem__(self, idx):
        Real = self.reals[idx]
        synthsis = self.synthsis[idx]
        label = self.label_list[idx]

        Real = harmonize_mr(Real)
        synthsis = harmonize_mr(synthsis)
        Real = torch.from_numpy(Real).unsqueeze(0).float()
        synthsis = torch.from_numpy(synthsis).unsqueeze(0).float()
        if self.aug:
            # random Horizontal and Vertical Flip to both image and mask
            p1 = np.random.choice([0, 1])
            p2 = np.random.choice([0, 1])
            self.trans = transforms.Compose([
                transforms.RandomHorizontalFlip(p1),
                transforms.RandomVerticalFlip(p2),
                transforms.RandomRotation(10, resample=False, expand=False, center=None),
            ])
            Real = self.trans(Real)
            synthsis = self.trans(synthsis)
        # Convert data to PyTorch tensor

        data = torch.cat((Real, synthsis), dim=0)
        label = torch.tensor(label).unsqueeze(0).float()
        return data, label"""

def harmonize_mr(X):#鐩稿
    X[X > 255] = 255
    X = X / 255 * 2
    return X


def test_pred_3D(net, Real, Fake, Fake_reverse, ROI, opt):
    pred_hope1 = torch.zeros(1).cuda()
    pred_list_hope1 = torch.zeros(1).cuda()
    pred_hope0 = torch.zeros(1).cuda()
    pred_list_hope0 = torch.zeros(1).cuda()
    """
    pred_hope1 = torch.zeros(1)
    pred_list_hope1 = torch.zeros(1)
    pred_hope0 = torch.zeros(1)
    pred_list_hope0 = torch.zeros(1)"""
    if isinstance(Real, dict):
        Real['t1'] = harmonize_mr(Real['t1'])
        Real['t2f'] = harmonize_mr(Real['t2f'])
        Real['t1c'] = harmonize_mr(Real['t1c'])
    else:
        Real = harmonize_mr(Real)
    Fake = harmonize_mr(Fake)
    Fake_reverse = harmonize_mr(Fake_reverse)
    ROI[ROI != 0] = 1
    mask_size, mask_point = mask_decoder(ROI, '')
    Real, Fake, Fake_reverse = random_crop_with_mask_three(Real, Fake, Fake_reverse, mask_point, mask_size, [opt.depthSize, opt.ImageSize, opt.ImageSize], tain=False)

    pred_hope1 = torch.sigmoid(net(torch.from_numpy(np.concatenate((Real[np.newaxis, np.newaxis,...], Fake[np.newaxis, np.newaxis,...]), axis=1)).float().cuda()))
    pred_hope0 = torch.sigmoid(net(torch.from_numpy(np.concatenate((Real[np.newaxis, np.newaxis,...], Fake_reverse[np.newaxis, np.newaxis,...]), axis=1)).float().cuda()))

    return (pred_hope1, pred_hope0), (1, 0)


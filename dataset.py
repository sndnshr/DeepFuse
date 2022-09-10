import os
import random
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from torch.utils.data import Dataset

class MEFdataset(Dataset):
    def __init__(self, data_path, transform, phase) -> None:
        super(MEFdataset, self).__init__()
        self.dir_prefix = data_path
        self.hr_over = os.listdir(self.dir_prefix + 'HR_over/')
        self.hr_over.sort()
        self.hr_under = os.listdir(self.dir_prefix + 'HR_under/')
        self.hr_under.sort()
        self.hr = os.listdir(self.dir_prefix + 'GT/')
        self.hr.sort()

        self.phase = phase
        self.transform = transform
        self.patch_size = 64
        

    def __getitem__(self, idx):

        hr_over = cv2.imread(self.dir_prefix + 'HR_over/' + self.hr_over[idx])
        hr_over = cv2.cvtColor(hr_over, cv2.COLOR_BGR2RGB)

        hr_under = cv2.imread(self.dir_prefix + 'HR_under/' + self.hr_under[idx])
        hr_under = cv2.cvtColor(hr_under, cv2.COLOR_BGR2RGB)

        hr = cv2.imread(self.dir_prefix + 'GT/' + self.hr[idx])
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)


        random.seed(42)
        transform = A.Compose(
            [
                # A.Resize(256, 256),
                A.SmallestMaxSize(max_size=512, interpolation=1, always_apply=False, p=1),
                A.RandomCrop(height=self.patch_size, width=self.patch_size, always_apply=False, p=1),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ],
            additional_targets={'image0': 'image', 'image1': 'image'}
        )

        if self.phase == 'train':

            transformed = transform(image=hr_over, image0=hr_under, image1=hr)

            if self.transform:
                hr_over_p = self.transform(transformed['image'])
                hr_under_p = self.transform(transformed['image0'])
                hr_p = self.transform(transformed['image1'])

        else:
            hr_over_p, hr_under_p, hr_p = hr_over, hr_under, hr

            if self.transform:
                hr_over_p = self.transform(hr_over_p)
                hr_under_p = self.transform(hr_under_p)
                hr_p = self.transform(hr_p)


        sample = {'hr_over': hr_over_p, 'hr_under': hr_under_p, 'hr': hr_p}
        return sample
    

    def get_patch(self, h_over, h_under, h):

        lh, lw = np.asarray(h_over).shape[:2]

        h_stride = self.patch_size
        # print('lw=',lw, " ", lw - h_stride)
        # print('lh=',lh, " ", lh - h_stride)
        # print(np.asarray(h_over).shape)
        x = random.randint(0, lw - h_stride)
        y = random.randint(0, lh - h_stride)

        h_over_p = np.asarray(h_over)[y:y + h_stride, x:x + h_stride, :]
        h_under_p = np.asarray(h_under)[y:y + h_stride, x:x + h_stride, :]
        h_p = np.asarray(h)[y:y + h_stride, x:x + h_stride, :]

        return h_over_p, h_under_p, h_p


    def __len__(self):
        return len(self.hr)
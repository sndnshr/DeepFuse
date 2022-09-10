import os
import cv2
import time
import torch
import torch.nn
import numpy as np
from tqdm import trange
from torchvision import transforms, utils

from batch_transformers import BatchToTensor, BatchRGBToYCbCr, YCbCrToRGB
from dataset import MEFdataset
from model import DeepFuse
from mefssim import MEF_MSSSIM

EPS = 1e-8

class Test:
    def __init__(self, config):
        self.dir_prefix = config.testset
        self.hr_over = os.listdir(self.dir_prefix + 'HR_over/')
        # self.hr_over.sort()
        self.hr_under = os.listdir(self.dir_prefix + 'HR_under/')
        # self.hr_under.sort()
        assert len(self.hr_over) == len(self.hr_under)
        self.num_imgs = len(self.hr_over)

        self.model = DeepFuse().cuda()
        self.state = torch.load(config.model_path + 'best_ep.pth')
        self.model.load_state_dict(self.state['model'])
        # self.model.load_state_dict(torch.load('checkpoint/DeepFuse-00199.pt')['state_dict'])

        self.transform = transforms.Compose([
            BatchToTensor(),
            BatchRGBToYCbCr()
        ])

        self.loss_fn = MEF_MSSSIM(is_lum=True)
        self.fused_img_path = config.fused_img_path
        self.use_cuda = config.use_cuda


    def test(self):
        self.model.eval()
        with torch.no_grad():
            for idx in self.hr_over:
                print(idx)
                img1 = cv2.imread(self.dir_prefix + 'HR_over/' + idx)
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                img1 = torch.unsqueeze(self.transform(img1), 0)

                img2 = cv2.imread(self.dir_prefix + 'HR_under/' + idx)
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                img2 = torch.unsqueeze(self.transform(img2), 0)

                assert img1.shape == img2.shape
                save_name = os.path.splitext(os.path.split(idx)[1])[0]

                im_over = torch.squeeze(img1, dim=0)
                im_under = torch.squeeze(img2, dim=0)

                im_exp = torch.stack((im_under, im_over), dim=0)

                Ys = im_exp[:, 0, :, :].unsqueeze(1)
                Cbs = im_exp[:, 1, :, :].unsqueeze(1)
                Crs = im_exp[:, 2, :, :].unsqueeze(1)

                Wb = (torch.abs(Cbs - 0.5) + EPS) / torch.sum(torch.abs(Cbs - 0.5) + EPS, dim=0)
                Wr = (torch.abs(Crs - 0.5) + EPS) / torch.sum(torch.abs(Crs - 0.5) + EPS, dim=0)

                Cb_f = torch.sum(Wb * Cbs, dim=0, keepdim=True).clamp(0, 1)
                Cr_f = torch.sum(Wr * Crs, dim=0, keepdim=True).clamp(0, 1)

                Y_under = im_under[0, :, :].unsqueeze(0)
                Y_over = im_over[0, :, :].unsqueeze(0)

                if self.use_cuda:
                    Y_over = Y_over.cuda()
                    Y_under = Y_under.cuda()
                    Ys = Ys.cuda()

                Y_over = Y_over.unsqueeze(0)
                Y_under = Y_under.unsqueeze(0)


                self.model.setInput(Y_under, Y_over)
                Yf = self.model.forward()

                q = self.loss_fn(Yf, Ys).cpu()
                print(save_name, " ", q.data.item())

                Yf_RGB = YCbCrToRGB()(torch.cat((Yf.cpu(), Cb_f, Cr_f), dim=1))
                self._save_image(Yf_RGB, self.fused_img_path, save_name)


    def _save_image(self, image, path, name):
        b = image.size()[0]
        for i in range(b):
            t = image.data[i]
            t[t > 1] = 1
            t[t < 0] = 0
            utils.save_image(t, "%s/%s_%d.png" % (path, name, i))
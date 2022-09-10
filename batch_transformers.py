import numpy as np
from torchvision import transforms
import torch
import collections

class BatchToTensor(object):
    def __call__(self, img):

        return transforms.ToTensor()(img)

class BatchRGBToGray(object):
    def __call__(self, imgs):
        return [img[0, :, :] * 0.299 + img[1, :, :] * 0.587 + img[2:, :, :] * 0.114 for img in imgs]


# class BatchRGBToYCbCr(object):
#     def __call__(self, imgs):
#         return [torch.stack((0. / 256. + img[0, :, :] * 0.299000 + img[1, :, :] * 0.587000 + img[2, :, :] * 0.114000,
#                            128. / 256. - img[0, :, :] * 0.168736 - img[1, :, :] * 0.331264 + img[2, :, :] * 0.500000,
#                            128. / 256. + img[0, :, :] * 0.500000 - img[1, :, :] * 0.418688 - img[2, :, :] * 0.081312),
#                           dim=0) for img in imgs]

class BatchRGBToYCbCr(object):
    def __call__(self, img):
        return torch.stack((0. / 256. + img[0, :, :] * 0.299000 + img[1, :, :] * 0.587000 + img[2, :, :] * 0.114000,
                           128. / 256. - img[0, :, :] * 0.168736 - img[1, :, :] * 0.331264 + img[2, :, :] * 0.500000,
                           128. / 256. + img[0, :, :] * 0.500000 - img[1, :, :] * 0.418688 - img[2, :, :] * 0.081312),
                          dim=0)


class YCbCrToRGB(object):
    def __call__(self, img):
        return torch.stack((img[:, 0, :, :] + (img[:, 2, :, :] - 128 / 256.) * 1.402,
                            img[:, 0, :, :] - (img[:, 1, :, :] - 128 / 256.) * 0.344136 - (img[:, 2, :, :] - 128 / 256.) * 0.714136,
                            img[:, 0, :, :] + (img[:, 1, :, :] - 128 / 256.) * 1.772),
                            dim=1)

class RGBToYCbCr(object):
    def __call__(self, img):
        return torch.stack((0. / 256. + img[:, 0, :, :] * 0.299000 + img[:, 1, :, :] * 0.587000 + img[:, 2, :, :] * 0.114000,
                           128. / 256. - img[:, 0, :, :] * 0.168736 - img[:, 1, :, :] * 0.331264 + img[:, 2, :, :] * 0.500000,
                           128. / 256. + img[:, 0, :, :] * 0.500000 - img[:, 1, :, :] * 0.418688 - img[:, 2, :, :] * 0.081312),
                          dim=1)

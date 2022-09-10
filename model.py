import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels = 1, out_channels = 16, kernel_size = (5,5), padding = 2, factivation = nn.LeakyReLU) -> None:
        super().__init__()

        if kernel_size == (5,5):
            self.pad = 2
        elif kernel_size == (7,7):
            self.pad = 3

        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size,
                            stride=(1, 1),
                            padding=padding),
            
            torch.nn.BatchNorm2d(out_channels),

            factivation()
        )

    def forward(self, x):
        return self.layer(x)
      

# Layer to perform tensor addition Fm = F1 + F2
class FusionLayer(torch.nn.Module):
    def forward(self, F1, F2):
        return F1 + F2

class DeepFuse(torch.nn.Module):
    def __init__(self, device = 'cpu'):
        super().__init__()
        self.layer1 = ConvLayer(1, 16, 5)
        self.layer2 = ConvLayer(16, 32, 7, 3)

        self.layer3 = FusionLayer()
        self.layer4 = ConvLayer(32, 32, 7, 3)
        self.layer5 = ConvLayer(32, 16, 5)

        self.layer6 = ConvLayer(16, 1, 5, factivation = nn.Sigmoid)

        self.device = device
        self.to(self.device)

    def setInput(self, y_1, y_2):
        self.y_1 = y_1
        self.y_2 = y_2

    def forward(self):
        c11 = self.layer1(self.y_1)
        c12 = self.layer1(self.y_2)
        c21 = self.layer2(c11)
        c22 = self.layer2(c12)
        f_m = self.layer3(c21, c22)
        c3  = self.layer4(f_m)
        c4  = self.layer5(c3)
        c5  = self.layer6(c4)

        return c5
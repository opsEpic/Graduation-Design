import torch
import torch.nn as nn

from models.ecapa_tdnn.model import SEBlock, ASPBlock, Conv1dReluBn


class Res2Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, K, D, scale=8):
        super(Res2Conv1dReluBn, self).__init__()
        self.scale = scale

        self.layer0 = nn.ModuleList([
            Conv1dReluBn(in_channels // self.scale, out_channels // self.scale, K, D)
            for _ in range(self.scale - 1)
        ])
        self.layer1 = nn.ModuleList([
            Conv1dReluBn(in_channels // self.scale, out_channels // self.scale, 1, 1)
            for _ in range(self.scale - 1)
        ])
        self.bn = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        y = []
        for i, temp_x in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                temp_y = temp_x
            elif i == 1:
                temp_y = self.layer0[i - 1](temp_x) + self.layer1[i - 1](temp_x)
            else:
                temp_x = temp_x + y[i - 1]
                temp_y = self.layer0[i - 1](temp_x) + self.layer1[i - 1](temp_x)
            y.append(temp_y)

        x = torch.cat(y, dim=1)
        return self.bn(x)


class SERes2Block(nn.Module):
    def __init__(self, C, K, D):
        super(SERes2Block, self).__init__()

        self.layer0 = nn.Sequential(
            Conv1dReluBn(C, C, 1, 1),
            Res2Conv1dReluBn(C, C, K, D),
            Conv1dReluBn(C, C, 1, 1),
        )

        self.layer1 = SEBlock(C, C)

    def forward(self, x):
        x1 = self.layer0(x)
        return x + self.layer1(x1)


class PCFBlock(nn.Module):
    def __init__(self, C, K1, K2, D1, D2, N):
        super(PCFBlock, self).__init__()
        self.n = N

        self.layer0 = Conv1dReluBn(80 // N, C // N, K1, D1)
        self.layer1 = nn.Sequential(
            SERes2Block(C // N, K2, D2),
            SERes2Block(C // N, K2, D2)
        )

    def forward(self, x, x0=None):
        x = x.contiguous().view(x.shape[0] * self.n, -1, x.shape[-1])
        x = self.layer0(x)

        if x0 is not None:
            x0 = x0.view(x.shape[0], -1, x.shape[-1])
            x = x + x0

        x = self.layer1(x)
        return x.view(x.shape[0] // self.n, -1, x.shape[-1])


class PCFECAPATDNN(nn.Module):
    def __init__(self, C=512):
        super(PCFECAPATDNN, self).__init__()

        self.layer0 = PCFBlock(C, 5, 3, 1, 1, 8)
        self.layer1 = PCFBlock(C, 5, 3, 1, 2, 4)
        self.layer2 = PCFBlock(C, 5, 3, 1, 3, 2)
        self.layer3 = PCFBlock(C, 5, 3, 1, 4, 1)

        self.layer4 = nn.Sequential(
            Conv1dReluBn(4 * C, 1536, 1, 1),
            ASPBlock(1536),
            nn.Linear(2 * 1536, 192),
        )

    def forward(self, x):
        x1 = self.layer0(x)
        x2 = self.layer1(x, x1)
        x3 = self.layer2(x, x2)
        x4 = self.layer3(x, x3)

        x = self.layer4(torch.cat((x1, x2, x3, x4), dim=1))
        return nn.functional.normalize(x)

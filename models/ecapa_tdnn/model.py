import torch
import torch.nn as nn


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, K, D):
        super(Conv1dReluBn, self).__init__()

        self.layer0 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=K, dilation=D, padding=((K - 1) * D) // 2),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        return self.layer0(x)


class Res2Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, K, D, scale=8):
        super(Res2Conv1dReluBn, self).__init__()
        self.scale = scale

        self.layer0 = nn.ModuleList([
            Conv1dReluBn(in_channels // self.scale, out_channels // self.scale, K, D)
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
                temp_y = self.layer0[i - 1](temp_x)
            else:
                temp_y = self.layer0[i - 1](temp_x + y[i - 1])
            y.append(temp_y)

        x = torch.cat(y, dim=1)
        return self.bn(x)


class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck=128):
        super(SEBlock, self).__init__()

        self.layer0 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, bottleneck, kernel_size=1, dilation=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, out_channels, kernel_size=1, dilation=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.layer0(x)


class SERes2Block(nn.Module):
    def __init__(self, C, K, D):
        super(SERes2Block, self).__init__()

        self.layer0 = nn.Sequential(
            Conv1dReluBn(C, C, 1, 1),
            Res2Conv1dReluBn(C, C, K, D),
            Conv1dReluBn(C, C, 1, 1),
            SEBlock(C, C),
        )

    def forward(self, x):
        return x + self.layer0(x)


class ASPBlock(nn.Module):
    def __init__(self, in_channels, attention_channels=128):
        super(ASPBlock, self).__init__()

        self.layer0 = nn.Sequential(
            Conv1dReluBn(3 * in_channels, attention_channels, 1, 1),
            nn.Tanh(),
            Conv1dReluBn(attention_channels, in_channels, 1, 1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        global_x = torch.cat((
            x,
            torch.mean(x, dim=2, keepdim=True).repeat(1, 1, x.size()[-1]),
            torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, x.size()[-1]),
        ), 1)

        w = self.layer0(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        return torch.cat((mu, sg), 1)


class ECAPATDNN(nn.Module):
    def __init__(self, C=512):
        super(ECAPATDNN, self).__init__()

        self.layer0 = Conv1dReluBn(80, C, 5, 1)

        self.layer1 = SERes2Block(C, 3, 2)
        self.layer2 = SERes2Block(C, 3, 3)
        self.layer3 = SERes2Block(C, 3, 4)

        self.layer4 = nn.Sequential(
            Conv1dReluBn(3 * C, 1536, 1, 1),
            ASPBlock(1536),
            nn.Linear(2 * 1536, 192),
            nn.BatchNorm1d(192),
        )

    def forward(self, x):
        x = self.layer0(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        return self.layer4(torch.cat((x1, x2, x3), dim=1))

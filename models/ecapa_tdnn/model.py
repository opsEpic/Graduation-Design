import os

import torch
import torch.nn as nn
from torchaudio.transforms import MFCC
from torchviz import make_dot


class TDNN(nn.Module):
    def __init__(self, in_channels, out_channels, K, D):
        super(TDNN, self).__init__()

        self.layer0 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=K, dilation=D, padding=((K - 1) * D) // 2),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        return self.layer0(x)


class Res2_block(nn.Module):
    def __init__(self, in_channels, out_channels, K, D, scale=8):
        super(Res2_block, self).__init__()
        self.scale = scale

        self.layer0 = nn.ModuleList([
            TDNN(in_channels // self.scale, out_channels // self.scale, K, D)
            for _ in range(self.scale - 1)
        ])

        self.k = K
        self.d = D

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

        return torch.cat(y, dim=1)


class SE_block(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck=128):
        super(SE_block, self).__init__()

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


class SERes2_block(nn.Module):
    def __init__(self, C, K, D):
        super(SERes2_block, self).__init__()

        self.layer0 = nn.Sequential(
            TDNN(C, C, 1, 1),

            Res2_block(C, C, K, D),
            nn.ReLU(),
            nn.BatchNorm1d(C),

            TDNN(C, C, 1, 1),

            SE_block(C, C),
        )

    def forward(self, x):
        return x + self.layer0(x)


class AttentiveStatPooling_block(nn.Module):
    def __init__(self, in_channels, attention_channels=128):
        super(AttentiveStatPooling_block, self).__init__()

        self.layer0 = nn.Sequential(
            TDNN(3 * in_channels, attention_channels, 1, 1),
            nn.Tanh(),
            TDNN(attention_channels, in_channels, 1, 1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        t = x.size()[-1]
        global_x = torch.cat((
            x,
            torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
            torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t),
        ), dim=1)

        w = self.layer0(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
        x = torch.cat((
            mu,
            sg,
        ), dim=1)

        return x


class ECAPATDNN(nn.Module):
    def __init__(self, C, S):
        super(ECAPATDNN, self).__init__()

        self.layer0 = TDNN(80, C, 5, 1)

        self.layer1 = SERes2_block(C, 3, 2)
        self.layer2 = SERes2_block(C, 3, 3)
        self.layer3 = SERes2_block(C, 3, 4)

        self.layer4 = nn.Sequential(
            nn.Conv1d(3 * C, 3 * C, kernel_size=1),
            nn.ReLU(),

            AttentiveStatPooling_block(3 * C),
            nn.BatchNorm1d(3072),

            nn.Linear(3072, S),
            nn.BatchNorm1d(S),
        )

    def forward(self, x):
        x = self.layer0(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        return x


class ECAPATDNN_model:
    def __load__(self, path):
        models = []

        path0 = path
        for file0 in os.listdir(path0):
            if file0.endswith('.pt'):
                models.append(file0)

        if len(models) > 0:
            self.step = int(models[-1][:-3])
            self.model = torch.load(os.path.join(path0, models[-1]))

    def __init__(self, model_path, device, C=512, S=192):
        self.model_path = model_path
        self.step = 0

        self.mfcc = MFCC(16000, 80, melkwargs={
            'n_mels': 80,
            'n_fft': 512,
            'win_length': 400,
            'hop_length': 160,
            'f_min': 20,
            'f_max': 7600,
            'window_fn': torch.hamming_window,
        }).to(device)

        self.model = None
        self.__load__(model_path)
        if self.model is None:
            self.model = ECAPATDNN(C, S)
        self.model = self.model.to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def save(self) -> None:
        torch.save(self.model, self.model_path + "/%04d.pt" % self.step)

    def train(self, data) -> (float, int):
        self.model.train()

        batch_input, batch_expect = data[0], data[1]
        batch_input = self.mfcc(batch_input)

        batch_output = self.model(batch_input)
        loss = self.criterion(batch_output, batch_expect)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += 1
        return 0., 0

    def eval(self, data) -> (float, int):
        return self.test(data)

    def test(self, data) -> (float, int):
        self.model.eval()

        with torch.no_grad():
            batch_input, batch_expect = data[0], data[1]
            batch_input = self.mfcc(batch_input)

            batch_output = self.model(batch_input)
            loss = self.criterion(batch_output, batch_expect)
            _, batch_label = torch.max(batch_output.data, 1)
            correct = (batch_label == batch_expect).sum().item()

            return loss.item(), correct

    def draw(self, data) -> None:
        self.model.train()

        batch_input, batch_expect = data[0], data[1]
        batch_input = self.mfcc(batch_input)

        y = self.model(batch_input)
        make_dot(y.mean(), params=dict(self.model.named_parameters())).view()

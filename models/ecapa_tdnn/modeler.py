import os

import torch
import torch.nn as nn
import torchaudio
from torchviz import make_dot

from models.ecapa_tdnn.model import ECAPATDNN
from models.ecapa_tdnn.loss import AAMsoftmax


class MFCC(nn.Module):
    def __init__(self):
        super(MFCC, self).__init__()

        self.layer0 = torchaudio.transforms.MFCC(16000, 80, melkwargs={
            'n_mels': 80,
            'n_fft': 512,
            'win_length': 400,
            'hop_length': 160,
            'f_min': 20,
            'f_max': 7600,
            'window_fn': torch.hamming_window,
        })

    def forward(self, x):
        with torch.no_grad():
            y = torch.nn.functional.pad(x, (1, 0), 'reflect')[:, :-1]
            y = self.layer0(x - 0.97 * y)
            return y - torch.mean(y, dim=1, keepdim=True)


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

        self.mfcc = MFCC().to(device)

        self.model = None
        self.__load__(model_path)
        if self.model is None:
            self.model = ECAPATDNN(C)
        self.model = self.model.to(device)

        self.criterion = AAMsoftmax(S, 0.2, 30).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def save(self) -> None:
        torch.save(self.model, self.model_path + "/%04d.pt" % self.step)

    def train(self, data) -> float:
        self.model.train()

        batch_input, batch_expect = data[0], data[1]
        batch_input = self.mfcc(batch_input)

        batch_output = self.model(batch_input)
        loss = self.criterion(batch_output, batch_expect)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += 1
        return loss.item()

    def eval(self, data) -> float:
        self.model.eval()

        with torch.no_grad():
            batch_input, batch_expect = data[0], data[1]
            batch_input = self.mfcc(batch_input)

            batch_output = self.model(batch_input)
            loss = self.criterion(batch_output, batch_expect)

            return loss.item()

    def test(self, audio1, audio2, label) -> (int, int, int, int):
        true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0

        if label == self.infer(audio1, audio2):
            if label:
                true_positive += 1
            else:
                true_negative += 1
        else:
            if label:
                false_positive += 1
            else:
                false_negative += 1

        return true_positive, true_negative, false_positive, false_negative

    def infer(self, audio1, audio2) -> int:
        self.model.eval()

        with torch.no_grad():
            vec1 = self.model(self.mfcc(audio1.view(1, -1)))
            vec2 = self.model(self.mfcc(audio2.view(1, -1)))

            score = torch.sum(vec1 * vec2).tolist()
            return 1 if score >= 0.5 else 0

    def draw(self, data) -> None:
        self.model.train()

        batch_input = data[0]
        batch_input = self.mfcc(batch_input)

        make_dot(self.model(batch_input).mean(), params=dict(self.model.named_parameters())).view()

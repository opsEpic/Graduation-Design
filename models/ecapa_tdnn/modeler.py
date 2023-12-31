import os

import torch
from torchaudio.transforms import MFCC
from torchviz import make_dot

from models.ecapa_tdnn.model import ECAPATDNN
from models.ecapa_tdnn.loss import AAMsoftmax


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

        self.criterion = AAMsoftmax(0.2, 30).to(device)
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
        self.model.eval()

        with torch.no_grad():
            batch_input, batch_expect = data[0], data[1]
            batch_input = self.mfcc(batch_input)

            batch_output = self.model(batch_input)
            loss = self.criterion(batch_output, batch_expect)

            return loss.item(), 0

    def test(self, data) -> (float, int):
        self.model.eval()

        with torch.no_grad():
            batch_input, batch_expect = data[0], data[1]
            batch_input = self.mfcc(batch_input)

            batch_output = self.model(batch_input)

            _, batch_label = torch.max(batch_output.data, 1)
            correct = (batch_label == batch_expect).sum().item()

            return 0., correct

    def infer(self, data) -> int:
        self.model.eval()

        with torch.no_grad():
            batch_input = data[0]
            batch_input = self.mfcc(batch_input)

            batch_output = self.model(batch_input)

            _, batch_label = torch.max(batch_output.data, 1)
            label = torch.mode(batch_label)

            return label.values.tolist()

    def draw(self, data) -> None:
        self.model.train()

        batch_input = data[0]
        batch_input = self.mfcc(batch_input)

        make_dot(self.model(batch_input).mean(), params=dict(self.model.named_parameters())).view()

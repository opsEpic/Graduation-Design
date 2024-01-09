import glob
import os

import torch
from torchviz import make_dot

from models.ecapa_tdnn.tools import MFCC
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

        self.preprocess = MFCC().to(device)
        self.model = ECAPATDNN(C).to(device)
        self.criterion = AAMsoftmax(S, 0.2, 30).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def search(self, filename):
        return glob.glob(filename + "*.pt", root_dir=self.model_path)

    def load(self, filename, step):
        self.model = torch.load(os.path.join(self.model_path, filename + "/%06d.pt" % step))
        self.step = step

    def save(self):
        torch.save(self.model, self.model_path + "/%06d.pt" % self.step)

    def train(self, audio, label) -> float:
        self.model.train()

        batch_input = self.preprocess(audio)

        batch_output = self.model(batch_input)
        loss = self.criterion(batch_output, label)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += 1
        return loss.item()

    def eval(self, audio1, audio2, label) -> (int, int, int, int):
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
            vec1 = self.model(self.preprocess(audio1.view(1, -1)))
            vec2 = self.model(self.preprocess(audio2.view(1, -1)))

            score = torch.sum(vec1 * vec2).tolist()
            return 1 if score >= 0.5 else 0

    def draw(self, audio) -> None:
        self.model.train()

        batch_input = self.preprocess(audio)

        make_dot(self.model(batch_input).mean(), params=dict(self.model.named_parameters())).view()

import os

import soundfile
import torch
import numpy
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F

from models.loss import AAMsoftmax
from models.tools import FbankAug, tuneThresholdfromScore


class Modeler(nn.Module):
    def __init__(self, model, speaker, device, path=None):
        super(Modeler, self).__init__()

        self.epoch = 0

        self.preprocess = FbankAug()
        self.model = model
        self.criterion = AAMsoftmax(speaker)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.97)

        if path is not None:
            models = []

            path0 = path
            for file0 in os.listdir(path0):
                if file0.endswith('.pt'):
                    models.append(file0)

            if len(models) > 0:
                for name, param in torch.load(os.path.join(path0, models[-1])).items():
                    self.epoch = int(models[-1][:-3])
                    self.state_dict()[name].copy_(param)

        self.to(device)

    def model_train(self, loader):
        self.train()

        for audio, label in tqdm(loader):
            audio = audio.cuda()
            label = torch.LongTensor(label).cuda()

            batch_input = self.preprocess(audio, True)
            batch_output = self.model(batch_input)
            loss = self.criterion(batch_output, label)

            self.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()
        self.epoch += 1

    def model_eval(self, eval_list, eval_path):
        self.eval()

        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()

        for file in tqdm(setfiles):
            audio, _ = soundfile.read(os.path.join(eval_path, file))
            data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()

            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')

            feats = []
            startframe = numpy.linspace(0, audio.shape[0] - max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf) + max_audio])

            feats = numpy.stack(feats, axis=0).astype(numpy.float32)
            data_2 = torch.FloatTensor(feats).cuda()

            with torch.no_grad():
                embedding_1 = self.model(self.preprocess(data_1, False))
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = self.model(self.preprocess(data_2, False))
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]
        scores, labels = [], []

        for line in lines:
            embedding_11, embedding_12 = embeddings[line.split()[1]]
            embedding_21, embedding_22 = embeddings[line.split()[2]]

            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))

            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()

            scores.append(score)
            labels.append(int(line.split()[0]))

        return tuneThresholdfromScore(scores, labels, [1, 0.1])[1]

    def model_save(self, path):
        torch.save(self.state_dict(), path + "/%06d.pt" % self.epoch)

import math

import torch
import torch.nn as nn


class AAMsoftmax(nn.Module):
    def __init__(self, margin, scale):
        super(AAMsoftmax, self).__init__()

        self.margin = margin
        self.scale = scale

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, label):
        cosine = x
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale

        return self.ce(output, label)

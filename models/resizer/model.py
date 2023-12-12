import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

class Resizer(nn.Module):
    def __init__(self):
        super().__init__()

        self.upSample = nn.UpsamplingNearest2d(224)

    def forward(self, model_input):

        x = self.upSample(model_input)
        x = x.view(-1, 224, 224)
        return torch.stack((x, x, x), dim=1)


class Modeller:
    def __init__(self):
        self.model = Resizer()

    def run(self, data):
        return self.model(data)


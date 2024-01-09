import torch
import torch.nn as nn
import torchaudio


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
        y = torch.nn.functional.pad(x, (1, 0), 'reflect')[:, :-1]
        y = self.layer0(x - 0.97 * y)
        return y - torch.mean(y, dim=1, keepdim=True)

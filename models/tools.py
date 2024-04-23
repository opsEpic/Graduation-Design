import numpy
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from sklearn import metrics


def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr)))  # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE]) * 100.

    return tunedThreshold, eer, fpr, fnr


class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = .97):
        super().__init__()
        self.coef = coef
        self.register_buffer('flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0))

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class FbankAug(nn.Module):

    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        super().__init__()
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80),
        )

        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x, aug):
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug:
                x = self.mask_along_axis(x, dim=2)
                x = self.mask_along_axis(x, dim=1)
            return x

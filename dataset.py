import os
import random
import re

import numpy
import soundfile
import torch


class train_loader(object):
    def __init__(self, train_list, train_path, slice_length):
        self.slice_length = slice_length + 240

        self.file_list = []
        with open(train_list, 'r') as files:
            for line in files:
                pair = re.split(r'[* \n]+', line)
                pair[0] = int(pair[0])
                pair[1] = os.path.join(train_path, pair[1])

                self.file_list.append(pair)

    def __getitem__(self, index):
        audio, _ = soundfile.read(self.file_list[index][1])
        length = self.slice_length

        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)

        return torch.FloatTensor(audio[0]), self.file_list[index][0]

    def __len__(self):
        return len(self.file_list)

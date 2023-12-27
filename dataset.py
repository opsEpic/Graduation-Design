import os
import random
import re

import soundfile
import torch
from tqdm import tqdm


def creat_file_list(data_path, file_list_path):
    with open(file_list_path, 'w') as file:
        label = 0

        path0 = data_path
        for file0 in os.listdir(path0):
            path1 = os.path.join(path0, file0)

            for file1 in os.listdir(path1):
                path2 = os.path.join(path1, file1)

                for file2 in os.listdir(path2):
                    path3 = os.path.join(path2, file2)

                    file.write("{path}|{label}\n".format(path=path3, label=label))

            label += 1


def disrupt_file_list(file_list_path):
    file_list = []

    with open(file_list_path, 'r') as file:
        for line in file:
            file_list.append(line)

    for i in range(len(file_list)):
        rand = random.randint(i, len(file_list) - 1)
        file_list[i], file_list[rand] = file_list[rand], file_list[i]

    with open(file_list_path, 'w') as file:
        for line in file_list:
            file.write(line)


def separate_file_list(file_list_path, separate_file_list_path, separate_num):
    file_list = []

    with open(file_list_path, 'r') as file:
        for line in file:
            file_list.append(line)

    with open(separate_file_list_path, 'w') as file:
        for line in file_list[:separate_num]:
            file.write(line)

    with open(separate_file_list_path, 'w') as file:
        for line in file_list[:separate_num]:
            file.write(line)


class Dataset:
    def __load_file_list__(self, path):
        with open(path, 'r') as file_list:
            for line in file_list:
                pair = re.split(r'[*|]+', line)
                pair[1] = int(pair[1])

                self.file_list.append(pair)

    def __init__(self, path, batch_size, slice_length, if_tqdm, device):
        self.file_list = []
        self.file_list_pointer = 0
        self.__load_file_list__(path)

        self.data_batch_buffer = []

        self.batch_size = batch_size
        self.slice_length = slice_length
        self.device = device

        self.if_tqdm = if_tqdm
        self.tqdm = None

    def get_batch(self) -> (list, bool):  # batch, end
        def make_batch(data_batch):
            batch_input = []
            batch_expect = []
            for i in data_batch:
                batch_input.append(i[0])
                batch_expect.append(i[1])

            return [torch.stack(batch_input), torch.tensor(batch_expect, device=self.device)]

        if self.file_list_pointer == 0 and len(self.data_batch_buffer) == 0:
            if self.if_tqdm:
                self.tqdm = tqdm(total=len(self.file_list))

        while len(self.data_batch_buffer) <= self.batch_size:
            if self.file_list_pointer == 0 and len(self.data_batch_buffer) > 0:
                if self.if_tqdm:
                    self.tqdm.close()
                    self.tqdm = True

                ans = self.data_batch_buffer
                self.data_batch_buffer = []
                return make_batch(ans), True

            temp = self.file_list[self.file_list_pointer]
            self.file_list_pointer = (self.file_list_pointer + 1) % len(self.file_list)
            if self.if_tqdm:
                self.tqdm.update(1)

            audio = soundfile.read(temp[0], dtype='float32')
            audio = torch.from_numpy(audio[0]).to(self.device)
            audio /= torch.max(torch.abs(audio))

            while len(audio) >= self.slice_length // 2:
                if len(audio) < self.slice_length:
                    audio = torch.cat((audio, torch.zeros(self.slice_length - len(audio), dtype=torch.float32, device=self.device)))

                self.data_batch_buffer.append([audio[:self.slice_length], temp[1]])
                audio = audio[self.slice_length:]

        ans = self.data_batch_buffer[:self.batch_size]
        self.data_batch_buffer = self.data_batch_buffer[self.batch_size:]
        return make_batch(ans), False

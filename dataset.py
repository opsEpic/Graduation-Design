import asyncio
import glob
import os
import random
import re

import soundfile
import torch
from tqdm import tqdm


def creat_file_list(data_path, file_list_path):
    with open(file_list_path, 'w') as file:
        for i, file1 in enumerate(os.listdir(data_path)):
            for file2 in glob.glob(os.path.join(file1, '*/*.wav'), root_dir=data_path):
                file.write(f"{i} {file2}\n")


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

    with open(file_list_path, 'w') as file:
        for line in file_list[separate_num:]:
            file.write(line)

    with open(separate_file_list_path, 'w') as file:
        for line in file_list[:separate_num]:
            file.write(line)


def __read_norm__(path, device):
    audio = soundfile.read(path, dtype='float32')[0]
    audio = torch.from_numpy(audio).to(device)
    audio /= torch.max(torch.abs(audio))
    return audio


class Dataset:
    def __load_file_list__(self, dataset_path, file_list_path):
        with open(file_list_path, 'r') as file_list:
            for line in file_list:
                pair = re.split(r'[* \n]+', line)
                pair[0] = int(pair[0])
                pair[1] = os.path.join(dataset_path, pair[1])

                self.file_list.append(pair[:2])

    def __init__(self, dataset_path, file_list_path, batch_size, slice_length, device):
        self.file_list = []
        self.file_list_pointer = 0
        self.__load_file_list__(dataset_path, file_list_path)

        self.data_batch_buffer = []
        self.tqdm = None

        self.batch_size = batch_size
        self.slice_length = slice_length
        self.device = device

    def __make_batch__(self, data_batch):
        batch_input = []
        batch_expect = []
        for i in data_batch:
            batch_input.append(i[0])
            batch_expect.append(i[1])

        return [torch.stack(batch_input), torch.tensor(batch_expect, device=self.device)]

    def __get_batch__(self):
        while len(self.data_batch_buffer) < self.batch_size:
            if self.file_list_pointer >= len(self.file_list):
                if len(self.data_batch_buffer) > 0:
                    ans = self.data_batch_buffer
                    self.data_batch_buffer = []
                    return self.__make_batch__(ans)
                else:
                    return None

            pair = self.file_list[self.file_list_pointer]
            self.file_list_pointer += 1
            self.tqdm.update(1)

            audio = __read_norm__(pair[1], self.device)

            while len(audio) >= self.slice_length // 2:
                if len(audio) < self.slice_length:
                    audio = torch.cat((audio, torch.zeros(self.slice_length - len(audio), dtype=torch.float32, device=self.device)))

                self.data_batch_buffer.append([audio[:self.slice_length], pair[0]])
                audio = audio[self.slice_length:]

        ans = self.data_batch_buffer[:self.batch_size]
        self.data_batch_buffer = self.data_batch_buffer[self.batch_size:]
        return self.__make_batch__(ans)

    def __next__(self):
        ans = self.__get_batch__()

        if ans is None:
            self.file_list_pointer = 0
            self.tqdm.close()
            raise StopIteration

        return ans

    def __iter__(self):
        self.tqdm = tqdm(total=self.__len__())
        return self

    def __len__(self):
        return len(self.file_list)


class Dataset_test:
    def __load_file_list__(self, dataset_path, file_list_path):
        with open(file_list_path, 'r') as file_list:
            for line in file_list:
                pair = re.split(r'[* \n]+', line)
                pair[0] = int(pair[0])
                pair[1] = os.path.join(dataset_path, pair[1])
                pair[2] = os.path.join(dataset_path, pair[2])

                self.file_list.append(pair[:3])

    def __init__(self, dataset_path, path, pair_num, slice_length, device):
        self.file_list = []
        self.file_list_pointer = 0
        self.__load_file_list__(dataset_path, path)

        self.file_list = self.file_list[:pair_num]
        self.tqdm = None

        self.slice_length = slice_length
        self.device = device

    def __get_batch__(self):
        if self.file_list_pointer >= len(self.file_list):
            return None

        pair = self.file_list[self.file_list_pointer]
        self.file_list_pointer += 1
        self.tqdm.update(1)

        audio1 = __read_norm__(pair[1], self.device)
        audio2 = __read_norm__(pair[2], self.device)

        return audio1, audio2, pair[0]

    def __next__(self):
        ans = self.__get_batch__()

        if ans is None:
            self.file_list_pointer = 0
            self.tqdm.close()
            raise StopIteration

        return ans

    def __iter__(self):
        self.tqdm = tqdm(total=self.__len__())
        return self

    def __len__(self):
        return len(self.file_list)

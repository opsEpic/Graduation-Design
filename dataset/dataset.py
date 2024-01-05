import glob
import os

import soundfile
import torch

from dataset.dataset_interface import Dataset_interface


def creat_file_list(data_path, file_list_path):
    with open(file_list_path, 'w') as file:
        for i, file1 in enumerate(os.listdir(data_path)):
            for file2 in glob.glob(os.path.join(file1, '*/*.wav'), root_dir=data_path):
                file.write(f"{i} {file2}\n")


def audio_norm(audio, device):
    audio = torch.from_numpy(audio).to(device)
    return audio / torch.max(torch.abs(audio))


class Dataset_train(Dataset_interface):
    def __init__(self, dataset_path, path, pair_num, batch_size, slice_length, device):
        super(Dataset_train, self).__init__(dataset_path, path, pair_num, batch_size, slice_length, device)

    def __get_batch__(self):
        while len(self.data_batch_buffer) < self.batch_size:
            if self.file_list_pointer >= len(self.file_list):
                if len(self.data_batch_buffer) > 0:
                    break
                else:
                    return None

            pair = self.file_list[self.file_list_pointer]
            self.file_list_pointer += 1
            self.tqdm.update(1)

            audio = soundfile.read(pair[1], dtype='float32')[0]
            audio = audio_norm(audio, self.device)

            while len(audio) >= self.slice_length // 2:
                if len(audio) < self.slice_length:
                    audio = torch.cat((audio, torch.zeros(self.slice_length - len(audio), dtype=torch.float32, device=self.device)))

                self.data_batch_buffer.append([audio[:self.slice_length], pair[0]])
                audio = audio[self.slice_length:]

        ans = self.data_batch_buffer[:self.batch_size]
        self.data_batch_buffer = self.data_batch_buffer[self.batch_size:]

        batch_input = []
        batch_expect = []
        for i in ans:
            batch_input.append(i[0])
            batch_expect.append(i[1])
        return torch.stack(batch_input), torch.tensor(batch_expect, device=self.device)


class Dataset_eval(Dataset_interface):
    def __init__(self, dataset_path, path, pair_num, batch_size, slice_length, device):
        super(Dataset_eval, self).__init__(dataset_path, path, pair_num, batch_size, slice_length, device)

    def __get_batch__(self):
        if self.file_list_pointer >= len(self.file_list):
            return None

        pair = self.file_list[self.file_list_pointer]
        self.file_list_pointer += 1
        self.tqdm.update(1)

        audio1 = soundfile.read(pair[1], dtype='float32')[0]
        audio1 = audio_norm(audio1, self.device)
        audio2 = soundfile.read(pair[2], dtype='float32')[0]
        audio2 = audio_norm(audio2, self.device)

        return audio1, audio2, pair[0]

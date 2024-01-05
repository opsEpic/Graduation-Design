import os
import random
import re

from tqdm import tqdm


def __disrupt_file_list__(file_list):
    for i in range(len(file_list)):
        rand = random.randint(i, len(file_list) - 1)
        file_list[i], file_list[rand] = file_list[rand], file_list[i]

    return file_list


class Dataset_interface:
    def __load_file_list__(self, dataset_path, file_list_path):
        with open(file_list_path, 'r') as file_list:
            for line in file_list:
                pair = re.split(r'[* \n]+', line)
                pair[0] = int(pair[0])
                for i in range(1, len(pair)):
                    pair[i] = os.path.join(dataset_path, pair[i])

                self.file_list.append(pair)

    def __init__(self, dataset_path, path, pair_num, batch_size, slice_length, device):
        self.file_list_pointer = 0
        self.file_list = []
        self.__load_file_list__(dataset_path, path)
        self.file_list = __disrupt_file_list__(self.file_list)
        if pair_num > 0:
            self.file_list = self.file_list[:pair_num]

        self.batch_size = batch_size
        self.data_batch_buffer = []
        self.tqdm = None

        self.slice_length = slice_length
        self.device = device

    def __get_batch__(self):
        pass

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

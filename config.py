import json
import os
import logging

config_preset = {
    'train_dataset_path': './raw/vox2/wav',
    'train_list_path': './filelists/train.txt',

    'eval_dataset_path': './raw/vox1/wav',
    'eval_list_path': './filelists/list_test_H_cleaned.txt',

    'speaker': 5994,
    'batch_size': 32,
    'slice_length': 2 * 16000,
    'device': 'cuda',

    'save_path': './exps',
}


class Config:
    def save(self):
        with open(self.config_path, 'w') as config_file:
            json.dump(self.config, config_file, indent=4)

    def __load__(self):
        if not os.path.exists(self.config_path):
            self.save()

            logging.warning('config has been firstly created.')
            return

        with open(self.config_path, 'r') as config_file:
            config = json.load(config_file)

            if self.config.keys() != config.keys():
                logging.warning('config has more or less keys than preset.')

            self.config = config

    def __init__(self, config_path):
        self.config_path = config_path
        self.config = config_preset

        self.__load__()

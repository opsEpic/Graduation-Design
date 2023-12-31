import time
import logging

import torch

from config import Config
from models.ecapa_tdnn.modeler import ECAPATDNN_model
from dataset import creat_file_list, disrupt_file_list, separate_file_list, Dataset

config_preset = {
    'dataset_path': './dataset/vox1/test/wav',
    'eval_list_size': 32,
    'test_list_size': 32,
    'train_list_path': './filelist/train.list',
    'eval_list_path': './filelist/eval.list',
    'test_list_path': './filelist/test.list',

    'C': 512,
    'speaker': 40,
    'batch_size': 64,
    'slice_length': 2 * 16000,
    'model_train_epoch': 1,
    'device': 'cpu',

    'model_save': False,
    'model_save_path': './exp',
}


class Trainer:
    def __init__(self):
        self.config = Config('./config/train.json', config_preset)

        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

        self.model = None

    def model_pretrain(self):
        creat_file_list(self.config.config['dataset_path'], self.config.config['train_list_path'])
        disrupt_file_list(self.config.config['train_list_path'])
        separate_file_list(self.config.config['train_list_path'], self.config.config['eval_list_path'], int(self.config.config['eval_list_size']))
        separate_file_list(self.config.config['train_list_path'], self.config.config['test_list_path'], int(self.config.config['test_list_size']))

    def load_dataset(self):
        self.train_dataset = Dataset(self.config.config['train_list_path'], int(self.config.config['batch_size']), int(self.config.config['slice_length']), True, self.config.config['device'])
        self.eval_dataset = Dataset(self.config.config['eval_list_path'], int(self.config.config['batch_size']), int(self.config.config['slice_length']), False, self.config.config['device'])
        self.test_dataset = Dataset(self.config.config['test_list_path'], int(self.config.config['batch_size']), int(self.config.config['slice_length']), False, self.config.config['device'])

    def load_model(self):
        self.model = ECAPATDNN_model(self.config.config['model_save_path'], self.config.config['device'], int(self.config.config['C']), int(self.config.config['speaker']))

    def __model_run__(self, mode):  # mode: train eval test
        dataset = {
            'train': self.train_dataset,
            'eval': self.eval_dataset,
            'test': self.test_dataset,
        }
        operator = {
            'train': self.model.train,
            'eval': self.model.eval,
            'test': self.model.test,
        }

        loss = 0.
        correct = 0
        total = 0
        end = False
        while not end:
            batch, end = dataset[mode].get_batch()
            batch_loss, batch_correct = operator[mode](batch)
            loss += batch_loss
            correct += batch_correct
            total += len(batch[0])

        time.sleep(0.5)
        return loss, correct, total

    def model_train(self):
        step = int(self.config.config['model_train_epoch'])

        for _ in range(step):
            self.__model_run__('train')
            time.sleep(0.5)

            loss, _, _ = self.__model_run__('eval')
            logging.info(f'loss = {loss}')
            time.sleep(0.5)

        if self.config.config['model_save']:
            self.model.save()

    def model_test(self):
        _, correct, total = self.__model_run__('test')
        logging.info(f'correct = {correct}/{total}')
        time.sleep(0.5)

    def model_infer(self, audio):
        audio = torch.tensor(audio, device=self.config.config['device'])
        audio = audio / torch.max(audio)
        audio = audio.view(1, -1)
        return self.model.infer([audio, None])

    def model_draw(self):
        batch_size = int(self.config.config['batch_size'])
        slice_length = int(self.config.config['slice_length'])
        draw_batch = torch.zeros([batch_size, slice_length], device=self.config.config['device'])

        self.model.draw([draw_batch, None])


if __name__ == '__main__':  # example
    logging.basicConfig(level=logging.INFO)

    pretrain = Trainer()
    pretrain.model_pretrain()

    model = Trainer()
    model.load_dataset()
    model.load_model()

    model.model_train()
    model.model_test()

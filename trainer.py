import time
import logging

import torch

from config import Config
from models.ecapa_tdnn.modeler import ECAPATDNN_model
from dataset import creat_file_list, disrupt_file_list, separate_file_list, Dataset, Dataset_test

config_preset = {
    'train_dataset_path': './dataset/vox2/wav',
    'eval_num': 32,
    'train_list_path': './filelist/train.list',
    'eval_list_path': './filelist/eval.list',

    'C': 512,

    'speaker': 40,
    'batch_size': 32,
    'slice_length': 2 * 16000,
    'model_train_epoch': 1,
    'device': 'cpu',

    'model_save': False,
    'model_save_path': './exp',

    'test_dataset_path': './dataset/vox1/wav',
    'test_num': 1000,
    'test_list_path': './filelist/test.list',
}


class Trainer:
    def __init__(self):
        self.config = Config('./config/train.json', config_preset)
        self.model = None

    def model_pretrain(self):
        creat_file_list(self.config.config['train_dataset_path'], self.config.config['train_list_path'])
        disrupt_file_list(self.config.config['train_list_path'])
        separate_file_list(self.config.config['train_list_path'], self.config.config['eval_list_path'], int(self.config.config['eval_num']))

    def load_model(self):
        self.model = ECAPATDNN_model(self.config.config['model_save_path'], self.config.config['device'], int(self.config.config['C']), int(self.config.config['speaker']))

    def model_train(self):
        train_dataset = Dataset(self.config.config['train_dataset_path'], self.config.config['train_list_path'], int(self.config.config['batch_size']), int(self.config.config['slice_length']), self.config.config['device'])
        eval_dataset = Dataset(self.config.config['train_dataset_path'], self.config.config['eval_list_path'], int(self.config.config['batch_size']), int(self.config.config['slice_length']), self.config.config['device'])

        step = int(self.config.config['model_train_epoch'])

        for _ in range(step):
            for batch in train_dataset:
                self.model.train(batch)
            time.sleep(0.5)

            loss = 0.
            for batch in eval_dataset:
                loss += self.model.eval(batch)
            logging.info(f'loss = {loss}')
            time.sleep(0.5)

        if self.config.config['model_save']:
            self.model.save()

    def model_test(self):
        true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
        test_dataset = Dataset_test(self.config.config['test_dataset_path'], self.config.config['test_list_path'], int(self.config.config['test_num']), int(self.config.config['slice_length']), self.config.config['device'])

        for audio1, audio2, label in test_dataset:
            tp, tn, fp, fn = self.model.test(audio1, audio2, label)
            true_positive += tp
            true_negative += tn
            false_positive += fp
            false_negative += fn

        tpr = true_positive / (true_positive + false_negative)
        fpr = false_positive / (false_positive + true_negative)

        eer = (fpr + 1. - tpr) / 2
        logging.info(f'eer = {eer}')

    def model_infer(self, audio1, audio2):
        audio1 = torch.tensor(audio1, device=self.config.config['device'])
        audio1 = audio1 / torch.max(audio1)

        audio2 = torch.tensor(audio2, device=self.config.config['device'])
        audio2 = audio2 / torch.max(audio2)
        return self.model.infer(audio1, audio2)

    def model_draw(self):
        batch_size = int(self.config.config['batch_size'])
        slice_length = int(self.config.config['slice_length'])
        draw_batch = torch.zeros([batch_size, slice_length], device=self.config.config['device'])

        self.model.draw([draw_batch, None])


if __name__ == '__main__':  # example
    logging.basicConfig(level=logging.INFO)

    model = Trainer()
    model.model_pretrain()
    model.load_model()

    model.model_train()
    model.model_test()

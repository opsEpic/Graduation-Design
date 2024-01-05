import time
import logging

from config import Config
from models.ecapa_tdnn.modeler import ECAPATDNN_model
from dataset.dataset import creat_file_list, audio_norm, Dataset_train, Dataset_eval


class Trainer:
    def __init__(self):
        self.config = Config('./config/train.json')
        self.model = None

    def model_pretrain(self):
        creat_file_list(self.config.config['train_dataset_path'], self.config.config['train_list_path'])

    def __load_model__(self):
        if self.model is None:
            self.model = ECAPATDNN_model(self.config.config['model_save_path'], self.config.config['device'], int(self.config.config['C']), int(self.config.config['speaker']))

    def __eval__(self, batch_iter):
        true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
        for audio1, audio2, label in batch_iter:
            tp, tn, fp, fn = self.model.eval(audio1, audio2, label)
            true_positive += tp
            true_negative += tn
            false_positive += fp
            false_negative += fn

        tpr = true_positive / (true_positive + false_negative)
        fpr = false_positive / (false_positive + true_negative)

        eer = (fpr + 1. - tpr) / 2
        logging.info(f'eer = {eer}')

    def model_train(self):
        self.__load_model__()

        train_dataset = Dataset_train(self.config.config['train_dataset_path'], self.config.config['train_list_path'], -1, int(self.config.config['batch_size']), int(self.config.config['slice_length']), self.config.config['device'])
        eval_dataset = Dataset_eval(self.config.config['eval_dataset_path'], self.config.config['eval_list_path'], int(self.config.config['eval_num']), 1, int(self.config.config['slice_length']), self.config.config['device'])

        for _ in range(int(self.config.config['model_train_epoch'])):
            loss = 0.
            for audio, label in train_dataset:
                loss += self.model.train(audio, label)
            logging.info(f'loss = {loss}')
            time.sleep(0.5)

            self.__eval__(eval_dataset)
            time.sleep(0.5)

        if self.config.config['model_save']:
            self.model.save()

    def model_test(self):
        self.__load_model__()

        test_dataset = Dataset_eval(self.config.config['test_dataset_path'], self.config.config['test_list_path'], int(self.config.config['test_num']), 1, int(self.config.config['slice_length']), self.config.config['device'])

        self.__eval__(test_dataset)
        time.sleep(0.5)

    def model_infer(self, audio1, audio2):
        self.__load_model__()

        audio1 = audio_norm(audio1, self.config.config['device'])
        audio2 = audio_norm(audio2, self.config.config['device'])
        return self.model.infer(audio1, audio2)

    def model_draw(self):
        self.__load_model__()

        train_dataset = Dataset_train(self.config.config['train_dataset_path'], self.config.config['train_list_path'], -1, int(self.config.config['batch_size']), int(self.config.config['slice_length']), self.config.config['device'])

        for audio, _ in train_dataset:
            self.model.draw(audio)
            return


if __name__ == '__main__':  # example
    logging.basicConfig(level=logging.INFO)

    model = Trainer()
    model.model_pretrain()

    model.model_train()
    model.model_test()

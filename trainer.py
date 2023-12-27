import time

from config import *
import models.ecapa_tdnn.model as ecapa_tdnn
from dataset import *

config_train = Config('./config/train.json', {
    'dataset_path': './dataset/vox1/dev/wav',

    'eval_list_size': '40',
    'test_list_size': '40',

    'train_list_path': './filelist/train.list',
    'eval_list_path': './filelist/eval.list',
    'test_list_path': './filelist/test.list',

    'batch_size': 64,
    'slice_length': 2 * 16000,
    'lr': 0.001,

    'max_memory(GB)': 16.0,
    'device': 'cpu',

    'model_save_epoch': 5,
    'model_save_path': './exp',
})


def model_pretrain():
    creat_file_list(config_train.config['dataset_path'], config_train.config['train_list_path'])
    disrupt_file_list(config_train.config['train_list_path'])
    separate_file_list(config_train.config['train_list_path'], config_train.config['eval_list_path'], int(config_train.config['eval_list_size']))
    separate_file_list(config_train.config['train_list_path'], config_train.config['test_list_path'], int(config_train.config['test_list_size']))


def model_train():
    model = ecapa_tdnn.Modeller(512, 40, config_train.config['model_save_path'], config_train.config['device'])

    train_dataset = Dataset(config_train.config['train_list_path'], int(config_train.config['batch_size']), config_train.config['slice_length'], True, config_train.config['device'])
    eval_dataset = Dataset(config_train.config['eval_list_path'], int(config_train.config['batch_size']), config_train.config['slice_length'], False, config_train.config['device'])

    step = int(config_train.config['model_save_epoch'])
    while True:
        for _ in range(step):
            train_end = False
            while not train_end:
                train_batch, train_end = train_dataset.get_batch()
                model.train(train_batch)

            loss = 0.
            eval_end = False
            while not eval_end:
                eval_batch, eval_end = eval_dataset.get_batch()
                loss += model.eval(eval_batch)

            info_log('loss = {loss}'.format(loss=loss))
            time.sleep(0.5)

        model.save()
        time.sleep(0.5)


def model_test():
    model = ecapa_tdnn.Modeller(512, 40, config_train.config['device'], config_train.config['model_save_path'])

    test_dataset = Dataset(config_train.config['test_list_path'], config_train.config['batch_size'], config_train.config['slice_length'], False, config_train.config['device'])

    test_end = False
    while not test_end:
        test_batch, test_end = test_dataset.get_batch()
        model.eval(test_batch)


if __name__ == '__main__':
    # model_pretrain()
    # model_train()
    pass

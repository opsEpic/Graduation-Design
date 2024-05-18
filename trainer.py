import os
import time

import torch

from config import Config
from dataset import train_loader
from models.ecapa_tdnn.model import ECAPATDNN
from models.pcf_ecapa_tdnn.model import PCFECAPATDNN
from models.modeler import Modeler


def __get_last_model__(path):
    models = []

    for file0 in os.listdir(path):
        if file0.endswith('.pt'):
            models.append(file0)
    if len(models) == 0:
        return None

    return os.path.join(path, models[-1])


config = Config('config/config.json')
last_model = __get_last_model__(config.config['save_path'])

trainloader = train_loader(config.config['train_list_path'], config.config['train_dataset_path'], config.config['slice_length'])
trainloader = torch.utils.data.DataLoader(trainloader, batch_size=config.config['batch_size'], shuffle=True, num_workers=8, drop_last=True)

if __name__ == '__main__':
    t1 = Modeler(PCFECAPATDNN(), config.config['speaker'], config.config['device'], last_model)
    for _ in range(8):
        t1.model_train(trainloader)
        t1.model_save(config.config['save_path'])
        print(t1.model_eval(config.config['eval_list_path'], config.config['eval_dataset_path'], config.config['slice_length']))
        time.sleep(1)

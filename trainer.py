import time

import torch

from config import Config
from dataset import train_loader
from models.pcf_ecapa_tdnn.model import PCFECAPATDNN
from models.modeler import Modeler

config = Config('config/config.json')

trainloader = train_loader(config.config['train_list_path'], config.config['train_dataset_path'], config.config['slice_length'])
trainloader = torch.utils.data.DataLoader(trainloader, batch_size=config.config['batch_size'], shuffle=True, num_workers=8, drop_last=True)

if __name__ == '__main__':
    t1 = Modeler(PCFECAPATDNN(), config.config['speaker'], config.config['device'], config.config['save_path'])
    for i in range(4):
        t1.model_train(trainloader)
        t1.model_save(config.config['save_path'])
        print(t1.model_eval(config.config['eval_list_path'], config.config['eval_dataset_path']))
        time.sleep(100)

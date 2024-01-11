from config import Config
from dataset.dataset import Dataset_train, Dataset_test
import models.ecapa_tdnn.modeler

config = Config('config/config.json')

train_dataset = Dataset_train(config.config['train_dataset_path'], config.config['train_list_path'], -1, int(config.config['batch_size']), int(config.config['slice_length']), config.config['device'])
test_dataset = Dataset_test(config.config['test_dataset_path'], config.config['test_list_path'], int(config.config['test_num']), 1, int(config.config['slice_length']), config.config['device'])


class Trainer:
    def __init__(self, modeler_type):
        self.modeller = modeler_type(config.config['speaker'], config.config['device'])

    def get_err(self):
        eers = self.modeller.model_train(train_dataset, test_dataset, config.config['epochs'], config.config['repeats'])

        total = 0.
        for eer in eers:
            total += min(eer)

        return total / len(eers)


if __name__ == '__main__':
    trainer1 = Trainer(models.ecapa_tdnn.modeler.Modeler)
    print(trainer1.get_err())
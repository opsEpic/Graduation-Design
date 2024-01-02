import json
import os
import logging


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

    def __init__(self, config_path, preset: dict):
        self.config_path = config_path
        self.config = preset

        self.__load__()

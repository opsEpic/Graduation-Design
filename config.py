import json
import os

from mylog import *


class Config:
    def __load__(self):
        if not os.path.exists(self.config_path):
            with open(self.config_path, 'w') as config_file:
                json.dump(self.config, config_file)

            warn_log('config has been firstly created.')
            return

        with open(self.config_path, 'r') as config_file:
            config = json.load(config_file)

            if self.config.keys() != config.keys():
                warn_log('config has more or less keys than preset.')

            self.config = config

    def __init__(self, config_path, preset: dict):
        self.config_path = config_path
        self.config = preset

        self.__load__()

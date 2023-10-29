import json


class ConfigReader:

    def __init__(self):
        self.path = "config.json"

    def get_config(self):
        with open(self.path) as file:
            config = json.load(file)
        return config

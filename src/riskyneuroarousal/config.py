import json
import os
from dataclasses import dataclass


@dataclass
class Config:
    def __init__(self, file_path: str = "../config.json", **kwargs):
        if not os.path.exists(file_path):
            file_path = "../../config.json"
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Config file not found, check path ;)")

        with open(file_path, "r") as file:
            data = json.load(file)
        for key, value in data.items():
            setattr(self, key, value)

        # add any additional keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

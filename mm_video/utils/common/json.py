# -*- coding: utf-8 -*-
# @Time    : 2023/2/18 06:03
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : json.py

import json
from typing import Union, List, Dict

from xopen import xopen


def load_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data: Union[List, Dict], filename: str, save_pretty: bool = False, **kwargs):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, **kwargs))
        else:
            json.dump(data, f, **kwargs)


class JsonlReader:
    def __init__(self, file: str, auto_reset: bool = False):
        """

        :param file:
        :param auto_reset: Whether to automatically reset the file pointer to the beginning when the end is reached.
        """
        self.jsonl_file = xopen(file, "r")
        self._auto_reset = auto_reset

    def __iter__(self):
        return self

    def __next__(self):
        line = self.jsonl_file.readline()
        if line:
            return json.loads(line)
        else:
            if self._auto_reset:
                self.jsonl_file.seek(0)
            raise StopIteration

    def to_list(self):
        return list(iter(self))

    def close(self):
        return self.jsonl_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class JsonlWriter:
    def __init__(self, file: str):
        self.jsonl_file = xopen(file, "a")

    def write(self, data, **kwargs):
        self.jsonl_file.write(json.dumps(data, **kwargs) + "\n")

    def flush(self):
        self.jsonl_file.flush()

    def close(self):
        self.jsonl_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

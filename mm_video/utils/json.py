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


def save_json(data: Union[List, Dict], filename: str, save_pretty: bool = False, sort_keys: bool = False):
    class MyEncoder(json.JSONEncoder):

        def default(self, obj):
            if isinstance(obj, bytes):  # bytes->str
                return str(obj, encoding='utf-8')
            return json.JSONEncoder.default(self, obj)

    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, cls=MyEncoder, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


class JsonlReader:
    def __init__(self, file: str):
        self.jsonl_file = xopen(file, "r")

    def __iter__(self):
        return self

    def __next__(self):
        line = self.jsonl_file.readline()
        if line:
            return json.loads(line)
        else:
            raise StopIteration

    def to_list(self):
        return list(iter(self))


class JsonlWriter:
    def __init__(self, file: str):
        self.jsonl_file = xopen(file, "a")

    def write(self, data, **kwargs):
        self.jsonl_file.write(json.dumps(data, **kwargs) + "\n")

    def flush(self):
        self.jsonl_file.flush()

    def close(self):
        self.jsonl_file.close()

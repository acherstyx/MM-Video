# -*- coding: utf-8 -*-
# @Time    : 7/6/24
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : debug.py

"""
This file contains some useful utils for debugging.
"""

__all__ = [
    "dump_return",
]

import json
from pathlib import Path


def dump_return(dump_file: str, keep_return: bool = True):
    """
    Decorator to dump the return value of a function to a JSON file.
    Note: The return value must be JSON-serializable.

    Example:
    @dump_return("output.json")
    def my_func():
        return {"a": 1, "b": 2}

    :param dump_file: Path to the output JSON file.
    :param keep_return: If False, return value will be surpassed.
    :return: The original function's return value, if keep_return is True.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            Path(dump_file).parent.mkdir(parents=True, exist_ok=True)
            with open(dump_file, "w") as f:
                json.dump(ret, f, indent=4)
            if keep_return:
                return ret

        return wrapper

    return decorator

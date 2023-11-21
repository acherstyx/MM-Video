# -*- coding: utf-8 -*-
# @Time    : 2022/11/12 22:32
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : logging.py

import datetime

from dataclasses import dataclass


@dataclass
class LogConfig:
    # info/metadata
    root: str = "log"  # log root
    project_name: str = "unnamed_project"  # this should be set to determine the output directory
    experiment_name: str = "unnamed_experiment"  # this should be set to determine the output directory


def get_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

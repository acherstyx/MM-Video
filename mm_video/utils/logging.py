# -*- coding: utf-8 -*-
# @Time    : 2022/11/12 22:32
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : logging.py

import datetime
import os
import logging
import colorlog

from torch import distributed as dist

from dataclasses import dataclass
from enum import Enum


class LogLevel(Enum):
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET


@dataclass
class LogConfig:
    # info/metadata
    project_name: str = "unnamed_project"  # this should be set to determine the output directory
    experiment_name: str = "unnamed_experiment"  # this should be set to determine the output directory

    # log config
    root: str = "log"
    output_dir: str = "${hydra:runtime.output_dir}"
    console_level: LogLevel = LogLevel.INFO
    console_colorful: bool = True


def get_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

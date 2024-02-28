# -*- coding: utf-8 -*-
# @Time    : 2022/11/15 00:45
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : __init__.py

from . import config, trainer, runner

import warnings

warnings.filterwarnings("default", category=DeprecationWarning, module="mm_video")

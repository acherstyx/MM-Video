# -*- coding: utf-8 -*-
# @Time    : 6/8/24
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : trl.py

from typing import Optional, Dict
from dataclasses import dataclass

import trl

from .transformers import TrainingArguments

"""
Patch TRL Config for trainers.
Currently only add DPOConfig, tested on version trl==0.9.3,0.9.4.
"""


@dataclass
class DPOConfig(TrainingArguments, trl.DPOConfig):
    loss_type: str = "sigmoid"

# -*- coding: utf-8 -*-
# @Time    : 4/29/24
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : peft.py
import peft

from dataclasses import dataclass
from typing import Optional, List

from ._common import DelayPostInit

__all__ = ["PeftConfig", "LoraConfig"]


@dataclass
class PeftConfig(DelayPostInit, peft.PeftConfig):
    base_model_name_or_path: Optional[str] = None
    revision: Optional[str] = None
    peft_type: Optional[peft.PeftType] = None
    task_type: Optional[peft.TaskType] = None


@dataclass
class LoraConfig(PeftConfig, peft.LoraConfig):
    target_modules: Optional[List[str]] = None
    layers_to_transform: Optional[List[int]] = None
    layers_pattern: Optional[List[str]] = None

# -*- coding: utf-8 -*-
# @Time    : 2022/11/13 02:00
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : optimizer.py

from dataclasses import dataclass
from typing import Any

from mm_video.config.registry import register_optimizer_config


@register_optimizer_config(name="Adam")
@dataclass
class AdamConf:
    _target_: str = "torch.optim.adam.Adam"
    lr: Any = 0.001
    betas: Any = (0.9, 0.999)
    eps: Any = 1e-08
    weight_decay: Any = 0
    amsgrad: Any = False

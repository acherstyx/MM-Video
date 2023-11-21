# -*- coding: utf-8 -*-
# @Time    : 10/23/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : scheduler.py


from dataclasses import dataclass
from omegaconf import MISSING

from mm_video.config.registry import register_scheduler_config


@register_scheduler_config("cosine")
@dataclass
class CosineSchedule:
    _target_: str = "transformers.get_cosine_schedule_with_warmup"
    num_warmup_steps: int = MISSING
    num_training_steps: int = MISSING
    num_cycles: float = 0.5
    last_epoch: int = -1

# -*- coding: utf-8 -*-
# @Time    : 5/6/24
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : _common.py
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig
from typing import Union, Any


@dataclass
class DelayPostInit:
    _run_post_init: bool = False

    def __post_init__(self):
        # Don't run post-init until ready to convert to TrainingArgs
        if self._run_post_init:
            super().__post_init__()


def run_post_init(config: Union[DictConfig, Any]):
    if OmegaConf.is_config(config) and hasattr(config, "_run_post_init"):
        config._run_post_init = True
        return OmegaConf.to_object(config)
    elif hasattr(config, "_run_post_init"):
        config._run_post_init = True
        config.__post_init__()
        return config
    else:
        return config

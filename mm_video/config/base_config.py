# -*- coding: utf-8 -*-
# @Time    : 10/11/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : base_config.py

from hydra.core.config_store import ConfigStore

from omegaconf import MISSING
from dataclasses import dataclass, field
from typing import List, Any, Optional

"""
Define the structure of base configuration for this template
"""

__all__ = ["BaseConfig"]


@dataclass
class SystemConfig:
    # deterministic
    deterministic: bool = True
    seed: int = 210222


@dataclass
class LogConfig:
    # info/metadata
    root: str = "log"
    project_name: str = "unnamed_project"  # this should be set to determine the output directory
    experiment_name: str = "unnamed_experiment"  # this should be set to determine the output directory


@dataclass
class BaseConfig:
    # Basic information configuration
    system: SystemConfig = field(default_factory=SystemConfig)
    log: LogConfig = field(default_factory=LogConfig)
    paths: Any = MISSING

    # Main component configuration
    dataset: Any = MISSING
    model: Any = MISSING
    meter: Optional[Any] = None

    # Pipeline configuration
    trainer: Any = MISSING
    runner: Any = MISSING


ConfigStore.instance().store(name="mm_video_structured_config", node=BaseConfig)

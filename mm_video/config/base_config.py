# -*- coding: utf-8 -*-
# @Time    : 10/11/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : base_config.py

import hydra
from hydra.core.config_store import ConfigStore

from omegaconf import OmegaConf, MISSING
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

    root: str = "log"  # log root
    project_name: str = "unnamed_project"  # this should be set to determine the output directory
    experiment_name: str = "unnamed_experiment"  # this should be set to determine the output directory


@dataclass
class BaseConfig:
    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"trainer": "Trainer"},
            {"runner": "Runner"}
        ]
    )

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


ConfigStore.instance().store(name="base_config", node=BaseConfig)

if __name__ == "__main__":
    @hydra.main(version_base=None, config_name="config", config_path="../../configs")
    def main(cfg: BaseConfig):
        print(OmegaConf.to_yaml(cfg))


    main()

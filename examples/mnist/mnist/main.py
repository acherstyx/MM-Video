# -*- coding: utf-8 -*-
# @Time    : 2024/2/18
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : main.py

import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None, config_name="config",
            config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs"))
def main(cfg: DictConfig) -> None:
    runner = instantiate(cfg.runner)
    runner.run(cfg)


if __name__ == '__main__':
    main()

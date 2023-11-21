# -*- coding: utf-8 -*-
# @Time    : 10/23/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : runner.py

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from dataclasses import dataclass
from typing import *

import os
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.utils import data

from mm_video.utils.logging import get_timestamp
from mm_video.utils.train_utils import manual_seed

from mm_video.config import BaseConfig, register_runner_config
from mm_video.data import DataLoaderConfig, build_loader
from mm_video.modeling import ModelBuilderConfig, build_model
from mm_video.trainer import BaseTrainerConfig, BaseTrainer
from mm_video.modeling.meter import Meter, DummyMeter

import logging

logger = logging.getLogger(__name__)

__all__ = ["Runner", "RunnerConfig", "main"]


class Runner:
    cfg: BaseConfig
    dataloader: Dict[str, data.DataLoader]
    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
    meter: Meter
    trainer: BaseTrainer = None

    def __init__(self, cfg: BaseConfig):
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))  # get RANK from environment
        manual_seed(cfg.system)

        self.cfg = cfg
        self.build_dataloader(cfg.data_loader)
        self.build_model(cfg.model_builder)
        self.build_optimizer(cfg.optimizer)
        self.build_scheduler(cfg.scheduler)
        self.build_meter(cfg.meter)
        self.build_trainer(cfg.trainer)

    def build_dataloader(self, data_loader_config: DataLoaderConfig):
        self.dataloader = build_loader(data_loader_config)  # split: data.DataLoader

    def build_model(self, model_builder_config: ModelBuilderConfig):
        self.model = build_model(model_builder_config)

    def build_optimizer(self, optimizer_config: DictConfig):
        self.optimizer: optim.Optimizer = instantiate(optimizer_config, _partial_=True)(params=self.model.parameters())

    def build_scheduler(self, scheduler_config: DictConfig):
        scheduler_callable = instantiate(scheduler_config, _partial_=True)
        if scheduler_callable is not None:
            self.scheduler = scheduler_callable(optimizer=self.optimizer)
        else:
            self.scheduler = None

    def build_meter(self, meter_config: DictConfig):
        self.meter: Meter = instantiate(meter_config)
        if self.meter is None:
            logger.info("Meter is not specified.")
            self.meter = DummyMeter()

    def build_trainer(self, trainer_config: BaseTrainerConfig):
        self.trainer = instantiate(trainer_config, _partial_=True)(
            dataloader=self.dataloader,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            meter=self.meter
        )

    def run(self):
        self.trainer.train()


@register_runner_config(name=f"{Runner.__qualname__}")
@dataclass
class RunnerConfig:
    _target_: str = f"{__name__}.{Runner.__qualname__}"


@hydra.main(version_base=None, config_name="config",
            config_path=f"{os.path.dirname(os.path.abspath(__file__))}/../../configs")
def main(cfg: BaseConfig):
    print(f"{get_timestamp()} => Run trainer")
    runner = instantiate(cfg.runner, _partial_=True)(cfg)
    runner.run()
    print(f"{get_timestamp()} => Finished!")

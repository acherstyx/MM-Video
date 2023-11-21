# -*- coding: utf-8 -*-
# @Time    : 2022/11/12 21:57
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : meter.py

import logging

import torch
import torch.nn as nn
import torch.distributed.fsdp as fsdp

from hydra.utils import instantiate
from dataclasses import dataclass
from enum import Enum
from omegaconf import MISSING
from typing import Any

from mm_video.utils.profile import Timer

__all__ = ["Parallelism", "ModelBuilderConfig", "build_model"]

logger = logging.getLogger(__name__)


class Parallelism(Enum):
    CPU = "CPU"  # Only use CPU
    DDP = "DDP"
    FSDP = "FSDP"


@dataclass
class ModelBuilderConfig:
    defaults = [
        {"model": MISSING}
    ]

    parallelism: Parallelism = Parallelism.DDP if torch.cuda.is_available() else Parallelism.CPU
    ddp_find_unused_parameters: bool = False

    model: Any = MISSING


def build_model(cfg: ModelBuilderConfig):
    with Timer("Building the model from the configuration..."):
        model = instantiate(cfg.model)

    # model parallelism
    logger.debug("Applying model parallelism...")
    if cfg.parallelism in (Parallelism.DDP, Parallelism.FSDP):
        logger.debug("Moving model to device: %s...", torch.cuda.current_device())
        model.cuda()
        logger.debug("Model is moved to device: %s", torch.cuda.current_device())
        if cfg.parallelism == Parallelism.DDP:
            logger.debug("Building DistributedDataParallel, check whether the program is hanging...")
            model = nn.parallel.DistributedDataParallel(
                model,
                find_unused_parameters=cfg.ddp_find_unused_parameters
            )
        elif cfg.parallelism == Parallelism.FSDP:
            logger.debug("Building FullyShardedDataParallel, check whether the program is hanging...")
            model = fsdp.FullyShardedDataParallel(
                model,
                cpu_offload=fsdp.CPUOffload(offload_params=True)
            )
        else:
            raise RuntimeError(f"Model parallelism '{cfg.parallelism}' is not supported!")
    elif cfg.parallelism == Parallelism.CPU:
        pass
    else:
        raise RuntimeError(f"Model parallelism '{cfg.parallelism}' is not supported!")
    logger.debug("Successfully applied model parallelism.")
    return model

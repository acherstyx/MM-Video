# -*- coding: utf-8 -*-
# @Time    : 2022/11/12 21:57
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : meter.py

import logging

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    CPUOffload
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy
)
from functools import partial

from hydra.utils import instantiate
from dataclasses import dataclass
from enum import Enum
from omegaconf import MISSING
from typing import Any, Optional, List

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
    fsdp_transformer_layer_cls_to_wrap: Optional[List[str]] = None

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

            # From Hugging Face Trainer
            auto_wrap_policy = None
            if cfg.fsdp_transformer_layer_cls_to_wrap is not None:
                transformer_cls_to_wrap = set()
                for layer_class in cfg.fsdp_transformer_layer_cls_to_wrap:
                    transformer_cls = get_module_class_from_name(model, layer_class)
                    if transformer_cls is None:
                        raise Exception("Could not find the transformer layer class to wrap in the model.")
                    else:
                        transformer_cls_to_wrap.add(transformer_cls)
                auto_wrap_policy = partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=transformer_cls_to_wrap
                )

            model = FullyShardedDataParallel(
                model,
                cpu_offload=CPUOffload(offload_params=True),
                auto_wrap_policy=auto_wrap_policy
            )
        else:
            raise RuntimeError(f"Model parallelism '{cfg.parallelism}' is not supported!")
    elif cfg.parallelism == Parallelism.CPU:
        pass
    else:
        raise RuntimeError(f"Model parallelism '{cfg.parallelism}' is not supported!")
    logger.debug("Successfully applied model parallelism.")
    return model


def get_module_class_from_name(module, name):
    """
    Gets a class from a module by its name.

    Args:
        module (`torch.nn.Module`): The module to get the class from.
        name (`str`): The name of the class.
    """
    modules_children = list(module.children())
    if module.__class__.__name__ == name:
        return module.__class__
    elif len(modules_children) == 0:
        return
    else:
        for child_module in modules_children:
            module_class = get_module_class_from_name(child_module, name)
            if module_class is not None:
                return module_class

# -*- coding: utf-8 -*-
# @Time    : 2022/11/12 21:46
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : meter.py

import torch.distributed as dist
from torch.utils import data

from mm_video.utils.train_utils import SystemConfig

from hydra.utils import instantiate
from dataclasses import dataclass
from omegaconf import MISSING
from typing import Optional, Dict, Any, Tuple

import logging

__all__ = ["DataLoaderConfig", "build_loader"]

logger = logging.getLogger(__name__)


@dataclass
class DataLoaderConfig:
    collate_fn: Optional[str] = None
    batch_size: int = 1
    num_workers: int = 0
    shuffle: bool = True
    prefetch_factor: Optional[int] = None
    multiprocessing_context: str = "spawn"
    system: SystemConfig = "${system}"  # TODO: simplify this

    dataset: Any = MISSING


def build_loader(
        cfg: DataLoaderConfig, split: Tuple[str, ...] = ("train", "test", "val")
) -> Dict[str, data.DataLoader]:
    logger.debug("Building dataloader...")
    assert type(split) is list or type(split) is tuple
    set_list = [instantiate(cfg.dataset, split=_split) for _split in split]
    if cfg.system.multiprocess:
        sampler_list = [data.distributed.DistributedSampler(dataset,
                                                            rank=dist.get_rank(),
                                                            shuffle=cfg.shuffle)
                        for _mode, dataset in zip(split, set_list)]
    else:
        sampler_list = [None for _mode in split]
    collate_fn = None
    kwargs_default = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": True,
        "persistent_workers": False,
        "shuffle": False if cfg.system.multiprocess else cfg.shuffle,
        "prefetch_factor": cfg.prefetch_factor,
        "collate_fn": collate_fn,
        "multiprocessing_context": cfg.multiprocessing_context if cfg.num_workers else None
    }
    loader_list = [data.DataLoader(dataset=_dataset, sampler=_sampler, **kwargs_default)
                   for _mode, _dataset, _sampler in zip(split, set_list, sampler_list)]

    logger.debug("Dataloader build finished.")
    if cfg.system.multiprocess and all(sampler is not None for sampler in sampler_list):
        res = {}
        res.update({_mode: loader for _mode, loader in zip(split, loader_list)})
        res.update({f"{_mode}_sampler": sampler for _mode, sampler in zip(split, sampler_list)})
        return res
    else:
        return {_mode: loader for _mode, loader in zip(split, loader_list)}

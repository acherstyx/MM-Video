# -*- coding: utf-8 -*-
# @Time    : 2022/11/12 21:46
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : meter.py

from hydra.utils import instantiate, get_object
from dataclasses import dataclass
from omegaconf import MISSING
from typing import Optional, Dict, Any, Tuple, Union

import os
from torch.utils import data

from mm_video.utils.profile import Timer

import logging

__all__ = ["DataLoaderConfig", "build_loader"]

logger = logging.getLogger(__name__)


@dataclass
class DataLoaderConfig:
    """DataLoader configuration options.

    Attributes:
        batch_size (int, optional): Batch size to use during training and evaluation, if not overriden.
            Defaults to 1.
        train_batch_size (int, optional): Batch size to use during training. Overrides `batch_size`, if set.
            Defaults to the value of `batch_size`.
        test_batch_size (int, optional): Batch size to use during testing. Overrides `batch_size`, if set.
            Defaults to the value of `batch_size`.
        eval_batch_size (int, optional): Batch size to use during evaluation. Overrides `batch_size`, if set.
            Defaults to the value of `batch_size`.
    """
    collate_fn: Optional[str] = None

    batch_size: int = 1
    train_batch_size: int = "${data_loader.batch_size}"
    test_batch_size: int = "${data_loader.batch_size}"
    eval_batch_size: int = "${data_loader.batch_size}"

    num_workers: int = 0
    shuffle: bool = True
    prefetch_factor: Optional[int] = None
    multiprocessing_context: str = "spawn"
    dataset: Any = MISSING


def build_loader(
        cfg: DataLoaderConfig, splits: Tuple[str, ...] = ("train", "test", "eval")
) -> Dict[str, Union[data.DataLoader, data.distributed.DistributedSampler]]:
    assert type(splits) is list or type(splits) is tuple
    assert all(split in ("train", "test", "eval") for split in splits), \
        f"Invalid split found in {splits}. Must be one of 'train', 'test', or 'eval'."
    timer = Timer(msg="Building dataloader...")
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    loader_and_sampler = {}
    for split in splits:
        dataset = instantiate(cfg.dataset, split=split)
        shuffle = cfg.shuffle if split == "train" else False
        batch_size = getattr(cfg, f"{split}_batch_size")
        collate_fn = get_object(cfg.collate_fn) if cfg.collate_fn is not None else None
        sampler = data.distributed.DistributedSampler(dataset, shuffle=shuffle) if world_size > 1 else None
        loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False if world_size > 1 else shuffle,
            sampler=sampler,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=cfg.prefetch_factor,
            multiprocessing_context=cfg.multiprocessing_context if cfg.num_workers else None
        )
        loader_and_sampler[split] = loader
        if sampler is not None:
            loader_and_sampler[f"{split}_sampler"] = sampler
    timer.end()
    return loader_and_sampler

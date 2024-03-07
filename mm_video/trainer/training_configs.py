# -*- coding: utf-8 -*-
# @Time    : 2024/3/5
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : training_configs.py
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, List

import torch
from torch.distributed.fsdp import ShardingStrategy

__all__ = ["TrainingStrategy", "TrainingConfig", "TrainingStrategyConfig", "DataLoaderConfig", "DebugConfig"]


class TrainingStrategy(Enum):
    cpu = CPU = "cpu"
    ddp = DDP = "ddp"
    fsdp = FSDP = "fsdp"


@dataclass
class TrainingConfig:
    num_train_epochs: int = 5

    amp: bool = False

    clip_norm: Optional[float] = None

    # Resume model from pretrained parameters
    resume: Optional[str] = None
    # Resume from last checkpoint saved in output directory
    resume_from_checkpoint: Union[bool, str] = False

    detect_anomaly: bool = False

    # Enable/disable PyTorch profiler
    write_profiler: bool = False
    # How often to write parameter histograms to TensorBoard. Set to `None` to disable.
    write_histogram: Optional[int] = None
    # How often to compute and write total gradient norm to TensorBoard. Set to `None` to disable.
    write_gradient_norm: Optional[int] = None

    save_epochs: Optional[int] = None
    save_steps: Optional[int] = None
    eval_steps: Optional[int] = None
    eval_epochs: Optional[int] = None
    logging_steps: Optional[int] = 500

    # Optimizer and Scheduler options
    learning_rate: float = 1e-5
    warmup_ratio: Optional[float] = None
    warmup_steps: Optional[int] = None

    gradient_accumulation_steps: int = 1


@dataclass
class TrainingStrategyConfig:
    strategy: TrainingStrategy = TrainingStrategy.ddp if torch.cuda.is_available() else TrainingStrategy.cpu

    # DDP options
    ddp_find_unused_parameters: bool = False

    # FSDP options
    fsdp_offload: bool = False
    fsdp_transformer_layer_cls_to_wrap: Optional[List[str]] = None
    fsdp_sync_module_states: bool = False
    fsdp_use_orig_params: bool = False
    fsdp_sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD


@dataclass
class DataLoaderConfig:
    """DataLoader configuration options.

    Attributes:
        train_batch_size (int): Batch size to use during training. Defaults to 1.
        eval_batch_size (int): Batch size to use during evaluation. Defaults to 1.
    """
    collate_fn: Optional[str] = None

    # batch size
    train_batch_size: int = 1
    eval_batch_size: int = 1

    num_workers: int = 0
    shuffle: bool = True
    prefetch_factor: Optional[int] = None
    multiprocessing_context: str = "spawn"
    pin_memory: bool = False
    persistent_workers: bool = False
    drop_last: bool = False


@dataclass
class DebugConfig:
    enable: bool = False
    save_inputs: bool = False
    save_inputs_for_each_step: bool = False

    max_train_steps: int = 10
    max_test_steps: int = 10
    max_eval_steps: int = 10

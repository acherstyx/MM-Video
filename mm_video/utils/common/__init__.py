# -*- coding: utf-8 -*-
# @Time    : 6/30/24
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : __init__.py

from .data import chunk
from .debug import dump_return
from .distributed import (
    get_rank,
    get_world_size,
    get_local_rank,
    get_local_world_size,
    get_master_addr,
    get_master_port,
    gather_object_multiple_gpus,
    conditional_gather_object_multiple_gpus,
    batch_gather_object_multiple_gpus,
    batch_conditional_gather_object_multiple_gpus,
)
from .json import (
    load_json,
    save_json,
    JsonlReader,
    JsonlWriter,
)
from .path import DisplayablePath
from .plot import (
    fig_to_image,
    fig_to_tensor,
    show_distribution,
)
from .registry import (
    Registry,
    LooseRegistry,
    PrefixRegistry,
    PostfixRegistry,
)
from .time import (
    format_time, get_timestamp, Timer
)
from .train_utils import (
    cuda,
    CudaPreFetcher,
    get_trainable_parameters,
    compute_total_gradient_norm,
    save_rng_state,
    load_rng_state,
)
from .writer import DummySummaryWriter, get_writer

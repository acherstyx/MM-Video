# -*- coding: utf-8 -*-
# @Time    : 2022/11/12 22:31
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : train_utils.py
import os
import math
import logging
import random
import numpy as np
import warnings
from typing import *

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel

logger = logging.getLogger(__name__)


def cuda(x: Union[torch.Tensor, Dict, List]):
    """
    Recursively move to GPU
    :param x:
    :return:
    """
    if isinstance(x, list) or isinstance(x, tuple):
        return [cuda(i) for i in x]
    elif isinstance(x, dict):
        return {k: cuda(v) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.cuda(non_blocking=True)
    else:
        return x


class CudaPreFetcher:
    def __init__(self, data_loader):
        self.dl = data_loader
        self.loader = iter(data_loader)
        self.stream = torch.cuda.Stream()
        self.batch = None

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            self.batch = self.cuda(self.batch)

    @staticmethod
    def cuda(x: Any):
        return cuda(x)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

    def __iter__(self):
        self.preload()
        return self

    def __len__(self):
        return len(self.dl)


def gather_object_multiple_gpu(*args, **kwargs):
    from .distributed import gather_object_multiple_gpus
    warnings.warn("gather_object_multiple_gpu from mm_video.utils.train_utils is deprecated, "
                  "please use gather_object_multiple_gpus from mm_video.utils.distributed instead",
                  DeprecationWarning, stacklevel=2)
    return gather_object_multiple_gpus(*args, **kwargs)


def conditional_gather_object_multiple_gpu(*args, **kwargs):
    from .distributed import conditional_gather_object_multiple_gpus
    warnings.warn("conditional_gather_object_multiple_gpu from mm_video.utils.train_utils is deprecated, "
                  "please use conditional_gather_object_multiple_gpus from mm_video.utils.distributed instead",
                  DeprecationWarning, stacklevel=2)
    return conditional_gather_object_multiple_gpus(*args, **kwargs)


# from peft: https://github.com/huggingface/peft/blob/main/src/peft/peft_model.py
def get_trainable_parameters(model: torch.nn.Module) -> Tuple[int, int, List[str]]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params = 0
    trainable_params_names = []
    all_param = 0
    for param_name, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
            trainable_params_names.append(param_name)

    return trainable_params, all_param, trainable_params_names


def compute_total_gradient_norm(model: nn.Module, norm_type: float = 2.0) -> torch.Tensor:
    """
    Compute gradient norm over all parameters.
    This needs to be called on all ranks since it uses collective communications.
    """
    from torch.distributed.fsdp.fully_sharded_data_parallel import _get_grad_norm

    norm_type = float(norm_type)
    if isinstance(model, FullyShardedDataParallel):
        # The logic for computing the total gradient norm for FSDP is adopted from the `clip_grad_norm_` method
        # of FullyShardedDataParallel

        # If every FSDP instance uses `NO_SHARD`, then we can directly use
        # the normal `_get_grad_norm` to calculate the total gradient norm
        if torch.__version__ < "2.1":
            import torch.distributed.fsdp._traversal_utils as traversal_utils
            all_handles = traversal_utils._get_fsdp_handles(model)
        else:
            all_handles = model._all_handles
        all_no_shard = all(
            not handle.uses_sharded_strategy for handle in all_handles
        )
        if all_no_shard:
            return _get_grad_norm(model.parameters(), norm_type=norm_type)
        sharded_params = set()
        nonsharded_params = set()  # `NO_SHARD` or not FSDP-managed
        grads: List[torch.Tensor] = []
        for handle in all_handles:
            target_set = (
                sharded_params if handle.uses_sharded_strategy else nonsharded_params
            )
            if handle._use_orig_params:
                for param in handle.flat_param._params:
                    target_set.add(param)
                    if param.grad is not None:
                        grads.append(param.grad)
            else:
                target_set.add(handle.flat_param)
                if handle.flat_param.grad is not None:
                    grads.append(handle.flat_param.grad)
        for param in model.parameters():
            not_fsdp_managed = (
                    param not in sharded_params and param not in nonsharded_params
            )
            if not_fsdp_managed:
                nonsharded_params.add(param)
                if param.grad is not None:
                    grads.append(param.grad)
        # Compute local norms (forced to be in FP32)
        local_sharded_norm = _get_grad_norm(sharded_params, norm_type).to(
            model.compute_device
        )
        local_nonsharded_norm = _get_grad_norm(nonsharded_params, norm_type).to(
            model.compute_device
        )
        # Reconstruct the total gradient norm depending on the norm type
        if norm_type == math.inf:
            total_norm = torch.maximum(local_sharded_norm, local_nonsharded_norm)
            dist.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.MAX, group=model.process_group
            )
        else:
            total_norm = local_sharded_norm ** norm_type
            dist.all_reduce(total_norm, group=model.process_group)
            # All-reducing the local non-sharded norm would count it an extra
            # world-size-many times
            total_norm += local_nonsharded_norm ** norm_type
            total_norm = total_norm ** (1.0 / norm_type)
        if model.cpu_offload.offload_params:
            total_norm = total_norm.cpu()
        return total_norm.to(torch.float32)
    else:
        parameters = [p for p in model.parameters() if p.grad is not None]
        return _get_grad_norm(parameters, norm_type=norm_type)


def save_rng_state(output_dir):
    """
    Based on `transformers.Trainer._save_rng_state`.

    :param output_dir:
    :return:
    """
    rng_states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        if dist.is_initialized():
            # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
            rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
        else:
            rng_states["cuda"] = torch.cuda.random.get_rng_state()

    os.makedirs(output_dir, exist_ok=True)

    if get_world_size() <= 1:
        torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
    else:
        torch.save(rng_states, os.path.join(output_dir, f"rng_state_{get_rank()}.pth"))


def load_rng_state(checkpoint):
    if checkpoint is None:
        return

    if get_world_size() <= 1:
        rng_file = os.path.join(checkpoint, "rng_state.pth")
        if not os.path.isfile(rng_file):
            logger.warning(
                "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                "fashion, reproducibility is not guaranteed."
            )
            return
    else:
        process_index = get_rank()
        rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
        if not os.path.isfile(rng_file):
            logger.warning(
                f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
                "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
            )
            return

    checkpoint_rng_state = torch.load(rng_file)
    random.setstate(checkpoint_rng_state["python"])
    np.random.set_state(checkpoint_rng_state["numpy"])
    torch.random.set_rng_state(checkpoint_rng_state["cpu"])
    if torch.cuda.is_available():
        if dist.is_initialized():
            torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
        else:
            try:
                torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
            except Exception as e:
                logger.warning(
                    f"Didn't manage to set back the RNG states of the GPU because of the following error:\n {e}"
                    "\nThis won't yield the same results as if the training had not been interrupted."
                )


def get_rank() -> int:
    """
    Get (global) rank from environment variable set by `torchrun`.
    """
    warnings.warn("get_rank is deprecated and will be moved to mm_video.utils.distributed",
                  DeprecationWarning, stacklevel=2)
    from .distributed import get_rank
    return get_rank()


def get_world_size() -> int:
    """
    Get (global) world size from environment variable set by `torchrun`.
    """
    warnings.warn("get_world_size is deprecated and will be moved to mm_video.utils.distributed",
                  DeprecationWarning, stacklevel=2)
    from .distributed import get_world_size
    return get_world_size()


def get_local_rank() -> int:
    warnings.warn("get_local_rank is deprecated and will be moved to mm_video.utils.distributed",
                  DeprecationWarning, stacklevel=2)
    from .distributed import get_local_rank
    return get_local_rank()


def get_local_world_size() -> int:
    warnings.warn("get_local_world_size is deprecated and will be moved to mm_video.utils.distributed",
                  DeprecationWarning, stacklevel=2)
    from .distributed import get_local_world_size
    return get_local_world_size()


def get_master_addr() -> str:
    warnings.warn("get_master_addr is deprecated and will be moved to mm_video.utils.distributed",
                  DeprecationWarning, stacklevel=2)
    from .distributed import get_master_addr
    return get_master_addr()


def get_master_port() -> Union[int, None]:
    warnings.warn("get_master_port is deprecated and will be moved to mm_video.utils.distributed",
                  DeprecationWarning, stacklevel=2)
    from .distributed import get_master_port
    return get_master_port()

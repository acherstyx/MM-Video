# -*- coding: utf-8 -*-
# @Time    : 2024/2/26
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : distributed.py
import os
import torch
import pickle
import hashlib
import itertools
import time
import torch.distributed as dist
from dataclasses import dataclass
import logging
from typing import *

logger = logging.getLogger(__name__)

__all__ = [
    "get_rank", "get_world_size", "get_local_rank", "get_local_world_size", "get_master_addr", "get_master_port",
    "gather_object_multiple_gpus", "conditional_gather_object_multiple_gpus"
]


def get_rank() -> int:
    """
    Get (global) rank from environment variable set by `torchrun`.
    """
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    """
    Get (global) world size from environment variable set by `torchrun`.
    """
    return int(os.environ.get("WORLD_SIZE", 1))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_local_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", 1))


def get_master_addr() -> str:
    return os.environ.get("MASTER_ADDR")


def get_master_port() -> Union[int, None]:
    port = os.environ.get("MASTER_PORT", None)
    if port is not None:
        return int(port)
    else:
        return None


def gather_object_multiple_gpus(list_object: List[Any], backend: AnyStr = "nccl", shared_folder=None,
                                retry=600, sleep=0.1):
    """
    gather a list of something from multiple GPU
    """
    assert type(list_object) == list, "`list_object` only receive list."
    assert backend in ["nccl", "filesystem"]
    if backend == "nccl":
        gathered_objects = [None for _ in range(dist.get_world_size())]
        logger.debug("Gathering with `all_gather_object`, please check if programme hanging...")
        dist.all_gather_object(gathered_objects, list_object)
        return list(itertools.chain(*gathered_objects))
    else:
        assert shared_folder is not None, "`share_folder` should be set if backend is `filesystem`"
        os.makedirs(shared_folder, exist_ok=True)
        uuid = torch.randint(99999999, 99999999999, size=(1,), dtype=torch.long).cuda()
        dist.all_reduce(uuid)
        uuid = hex(uuid.cpu().item())[-8:]
        with open(os.path.join(shared_folder, f"{uuid}_rank_{dist.get_rank():04d}.pkl"), "wb") as f:
            data = pickle.dumps(list_object)
            f.write(data)
        with open(os.path.join(shared_folder, f"{uuid}_rank_{dist.get_rank():04d}.md5"), "wb") as f:
            checksum = hashlib.md5(data).hexdigest()
            pickle.dump(checksum, f)
        gathered_list = []
        for rank in range(dist.get_world_size()):
            data_filename = os.path.join(shared_folder, f"{uuid}_rank_{rank:04d}.pkl")
            checksum_filename = os.path.join(shared_folder, f"{uuid}_rank_{rank:04d}.md5")
            data = None
            for _ in range(retry):
                time.sleep(sleep)
                try:
                    if not os.path.exists(data_filename):
                        continue
                    if not os.path.exists(checksum_filename):
                        continue
                    raw_data = open(data_filename, "rb").read()
                    checksum = pickle.load(open(checksum_filename, "rb"))
                    assert checksum == hashlib.md5(raw_data).hexdigest()
                    data = pickle.loads(raw_data)
                    break
                except Exception:
                    pass
            assert data is not None, f"Gather from filesystem failed after retry for {retry} times."
            gathered_list.extend(data)
        return gathered_list


def conditional_gather_object_multiple_gpus(
        list_object: List[Any],
        backend: AnyStr = "nccl", shared_folder=None, retry=600, sleep=0.1
):
    if dist.is_initialized():
        return gather_object_multiple_gpus(
            list_object=list_object,
            backend=backend,
            shared_folder=shared_folder,
            retry=retry,
            sleep=sleep
        )
    else:
        return list_object

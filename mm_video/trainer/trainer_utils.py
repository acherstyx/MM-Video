# -*- coding: utf-8 -*-
# @Time    : 2023/12/15
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : trainer_utils.py

import torch.distributed as dist

from typing import *
import logging

logger = logging.getLogger(__name__)


def barrier(debug_msg: Optional[str] = None):
    """
    A util for calling distributed barrier.

    @param debug_msg: Write message to debug log
    """
    if dist.is_initialized():
        if debug_msg is not None:
            logger.debug("Reached the '%s' barrier, waiting for other processes.", debug_msg)
        dist.barrier()
        if debug_msg is not None:
            logger.debug("Exited the '%s' barrier.", debug_msg)


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

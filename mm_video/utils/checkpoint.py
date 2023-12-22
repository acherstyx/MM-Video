# -*- coding: utf-8 -*-
# @Time    : 2022/11/13 00:25
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : checkpoint.py


import os
import torch
from torch import nn
from typing import List
import logging

logger = logging.getLogger(__name__)


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


def auto_resume(ckpt_folder: str):
    try:
        ckpt_files = [ckpt for ckpt in os.listdir(ckpt_folder) if ckpt.endswith(".pth")]
    except FileNotFoundError:
        ckpt_files = []
    if len(ckpt_files) > 0:
        return max([os.path.join(ckpt_folder, file) for file in ckpt_files], key=os.path.getmtime)
    else:
        return None


def save_model(model_file: str, model: torch.nn.Module):
    model = unwrap_model(model)
    torch.save(model.state_dict(), model_file)


def load_model(model_file: str, model: torch.nn.Module, strict=True):
    model = unwrap_model(model)
    state_dict = torch.load(model_file, map_location="cpu")

    missing_keys: List[str] = []
    unexpected_keys: List[str] = []
    error_msgs: List[str] = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        # mypy isn't aware that "_metadata" exists in state_dict
        state_dict._metadata = metadata  # type: ignore[attr-defined]

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model)
    del load

    if len(missing_keys) > 0:
        logger.info("Weights of {} not initialized from pretrained model: {}"
                    .format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)))
    if len(unexpected_keys) > 0:
        logger.info("Weights from pretrained model not used in {}: {}"
                    .format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
    if len(error_msgs) > 0:
        logger.info("Weights from pretrained model cause errors in {}: {}"
                    .format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))

    if len(missing_keys) == 0 and len(unexpected_keys) == 0 and len(error_msgs) == 0:
        logger.info("All keys loaded successfully for {}".format(model.__class__.__name__))

    if strict and len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
            model.__class__.__name__, "\n\t".join(error_msgs)))

# -*- coding: utf-8 -*-
# @Time    : 2024/3/15
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : training_arguments.py
import transformers
from packaging.version import Version
from transformers import TrainingArguments
from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Union, List

import logging

logger = logging.getLogger(__name__)

__all__ = ["HFTrainingArguments", "init_hf_training_args"]

"""
Patching HuggingFace transformers TrainingArguments to make it compatible with Hydra.

Currently, this patch is verified to be compatible with `transformers>=4.37.0,<=4.39.0`.
Other versions may also be supported, but not yet verified.
"""


@dataclass
class HFTrainingArguments(TrainingArguments):
    output_dir: str = "${hydra:runtime.output_dir}"
    _run_post_init: bool = False

    def __post_init__(self):
        # Don't run post-init until ready to convert to TrainingArgs
        if self._run_post_init:
            super().__post_init__()

    # Fix type
    debug: str = field(
        default="",
        metadata={
            "help": (
                "Whether or not to enable debug mode. Current options: "
                "`underflow_overflow` (Detect underflow and overflow in activations and weights), "
                "`tpu_metrics_debug` (print debug metrics on TPU)."
            )
        },
    )
    fsdp: str = field(
        default="",
        metadata={
            "help": (
                "Whether or not to use PyTorch Fully Sharded Data Parallel (FSDP) training (in distributed training"
                " only). The base option should be `full_shard`, `shard_grad_op` or `no_shard` and you can add"
                " CPU-offload to `full_shard` or `shard_grad_op` like this: full_shard offload` or `shard_grad_op"
                " offload`. You can add auto-wrap to `full_shard` or `shard_grad_op` with the same syntax: full_shard"
                " auto_wrap` or `shard_grad_op auto_wrap`."
            ),
        },
    )
    optim: str = field(
        default=TrainingArguments.default_optim,
        metadata={"help": "The optimizer to use."},
    )
    neftune_noise_alpha: Optional[float] = field(
        default=None,
        metadata={
            "help": "Activates neftune noise embeddings into the model. NEFTune has been proven to drastically improve"
                    " model performances for instrcution fine-tuning. Check out the original paper here: "
                    "https://arxiv.org/abs/2310.05914 and the original code here: https://github.com/neelsjain/NEFTune."
                    " Only supported for `PreTrainedModel` and `PeftModel` classes."
        },
    )

    # Change type to dict to allow configuring and overriding with hydra
    deepspeed: dict = field(
        default_factory=dict
    )


# Patch changes of TrainingArguments for each version
if Version(transformers.__version__) >= Version("4.38.0"):
    @dataclass
    class HFTrainingArguments(HFTrainingArguments):
        fsdp_config: Optional[str] = field(
            default=None,
            metadata={
                "help": (
                    "Config to be used with FSDP (Pytorch Fully Sharded  Data Parallel). The value is either a "
                    "fsdp json config file (e.g., `fsdp_config.json`) or an already loaded json file as `dict`."
                )
            },
        )
        optim_target_modules: Optional[List[str]] = field(
            default=None,
            metadata={
                "help": "Target modules for the optimizer defined in the `optim` argument. Only used for the GaLore"
                        " optimizer at the moment."
            },
        )


def init_hf_training_args(training_args: Union[DictConfig, TrainingArguments]) -> TrainingArguments:
    """
    Run __post_init__ for HFTrainingArguments.
    This function should be called before passing it to transformers Trainer.
    :param training_args:
    :return:
    """
    training_args._run_post_init = True
    return OmegaConf.to_object(training_args)

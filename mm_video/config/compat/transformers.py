# -*- coding: utf-8 -*-
# @Time    : 4/29/24
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : transformers.py
import transformers

from packaging.version import Version
from dataclasses import dataclass, field
from typing import Optional, List
from omegaconf import MISSING

import logging

from ._common import DelayPostInit

logger = logging.getLogger(__name__)

__all__ = ["TrainingArguments"]

"""
Patching HuggingFace transformers TrainingArguments to make it compatible with Hydra.

Currently, this patch is verified to be compatible with `transformers>=4.37.0,<=4.41.0`.
Other versions may also be supported, but not yet verified.
"""


@dataclass
class TrainingArguments(DelayPostInit, transformers.TrainingArguments):
    output_dir: str = MISSING

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
        default=transformers.TrainingArguments.default_optim,
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
    class TrainingArguments(TrainingArguments):
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

if Version(transformers.__version__) >= Version("4.41.0"):
    @dataclass
    class TrainingArguments(TrainingArguments):
        lr_scheduler_kwargs: dict = field(
            default_factory=dict,
            metadata={
                "help": (
                    "Extra parameters for the lr_scheduler such as {'num_cycles': 1} for the cosine with hard restarts."
                )
            },
        )
        accelerator_config: Optional[dict] = field(
            default=None,
            metadata={
                "help": (
                    "Config to be used with the internal Accelerator object initializtion. The value is either a "
                    "accelerator json config file (e.g., `accelerator_config.json`) or an already loaded json file as "
                    "`dict`."
                )
            },
        )
        report_to: Optional[List[str]] = field(
            default=None, metadata={"help": "The list of integrations to report the results and logs to."}
        )
        gradient_checkpointing_kwargs: Optional[dict] = field(
            default=None,
            metadata={
                "help": "Gradient checkpointing key word arguments such as `use_reentrant`. Will be passed to "
                        "`torch.utils.checkpoint.checkpoint` through `model.gradient_checkpointing_enable`."
            },
        )
        evaluation_strategy: Optional[str] = field(
            default=None,
            metadata={"help": "Deprecated. Use `eval_strategy` instead"},
        )

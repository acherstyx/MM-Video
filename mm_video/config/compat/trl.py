# -*- coding: utf-8 -*-
# @Time    : 6/8/24
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : trl.py

from typing import Optional, Dict

from .transformers import TrainingArguments
from dataclasses import dataclass

"""
Patch TRL Config for trainers.
Currently only add DPOConfig, tested on version trl==0.9.3,0.9.4.
"""


@dataclass
class DPOConfig(TrainingArguments):
    beta: float = 0.1
    label_smoothing: float = 0
    loss_type: str = "sigmoid"
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_target_length: Optional[int] = None
    is_encoder_decoder: Optional[bool] = None
    disable_dropout: bool = True
    generate_during_eval: bool = False
    precompute_ref_log_probs: bool = False
    dataset_num_proc: Optional[int] = None
    model_init_kwargs: Optional[Dict] = None
    ref_model_init_kwargs: Optional[Dict] = None
    model_adapter_name: Optional[str] = None
    ref_adapter_name: Optional[str] = None
    reference_free: bool = False
    force_use_ref_model: bool = False
    sync_ref_model: bool = False
    ref_model_mixup_alpha: float = 0.9
    ref_model_sync_steps: int = 64
    rpo_alpha: Optional[float] = None

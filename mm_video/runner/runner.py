# -*- coding: utf-8 -*-
# @Time    : 10/23/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : runner.py
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from typing import *

from torch.nn import Module
from torch.utils.data import Dataset

from mm_video.trainer.trainer_utils import manual_seed
from mm_video.config import BaseConfig, runner_store
from mm_video.modeling.meter import Meter
from mm_video.utils.common.time import Timer

import logging

logger = logging.getLogger(__name__)

__all__ = ["Runner"]


def is_target(x: Any) -> bool:
    if isinstance(x, dict):
        return "_target_" in x
    if OmegaConf.is_dict(x):
        return "_target_" in x
    return False


@runner_store
class Runner:
    """
    Runner is a basic entry point for building datasets and models, and running the training, testing, and evaluation
    loop.

    """

    def __init__(self, do_train: bool = False, do_test: bool = False, do_eval: bool = False):
        self.do_train = do_train
        self.do_test = do_test
        self.do_eval = do_eval
        # TODO: Add an option to load model state dict after build

    @staticmethod
    def build_dataset(
            dataset_config: DictConfig,
            with_train: bool = True, with_test: bool = True, with_eval: bool = True
    ) -> Dict[str, Dataset]:
        dataset_splits = []
        if with_train:
            dataset_splits.append("train")
        if with_test:
            dataset_splits.append("test")
        if with_eval:
            dataset_splits.append("eval")

        if is_target(dataset_config):
            with Timer("Building dataset from the configuration..."):
                dataset = {split: instantiate(dataset_config, split=split) for split in dataset_splits}
        elif (all(k in ("train", "test", "eval") for k in dataset_config.keys()) and
              all(is_target(v) or v is None for k, v in dataset_config.items())):
            for required_split in dataset_splits:
                if required_split not in dataset_config.keys():
                    raise ValueError(f"{required_split.capitalize()} is enabled while dataset is not provided."
                                     f"You must specify a dataset in config.")
            # Allow selecting different dataset for train, test and eval
            # See https://stackoverflow.com/a/71371396 for the config syntax
            # Example:
            # ```
            # defaults:
            #   - /dataset@dataset.train: TrainSet
            #   - /dataset@dataset.test: TestSet
            # ```
            dataset = {k: instantiate(v) for k, v in dataset_config.items() if v is not None}
        else:
            raise ValueError(f"Dataset config is invalid: \n{OmegaConf.to_yaml(dataset_config)}")
        return dataset

    @staticmethod
    def build_model(model_builder_config: DictConfig) -> Module:
        with Timer("Building model from the configuration..."):
            model = instantiate(model_builder_config)
        return model

    @staticmethod
    def build_meter(meter_config: DictConfig) -> Union[Meter, Dict[str, Meter], None]:
        if meter_config is None:
            return None
        elif is_target(meter_config):
            return instantiate(meter_config)
        elif all(is_target(v) for _, v in meter_config.items()):
            return {k: instantiate(v) for k, v in meter_config.items()}
        else:
            raise ValueError(f"Meter config is invalid: \n{OmegaConf.to_yaml(meter_config)}")

    def run(self, cfg: BaseConfig):
        if cfg.system.deterministic:
            manual_seed(cfg.system.seed)

        dataset = self.build_dataset(
            dataset_config=cfg.dataset,
            with_train=self.do_train,
            with_test=self.do_train and self.do_test,
            with_eval=self.do_eval,
        )
        model = self.build_model(cfg.model)
        meter = self.build_meter(cfg.meter)

        trainer = instantiate(cfg.trainer)(
            train_dataset=dataset["train"] if self.do_train else None,
            eval_dataset=dataset["test"] if self.do_train and self.do_test else None,
            model=model,
            meter=meter
        )

        if self.do_train:
            trainer.train()
        if self.do_eval:
            trainer.evaluate(dataset["eval"])

# -*- coding: utf-8 -*-
# @Time    : 10/23/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : runner.py

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from typing import *

import os
import shutil
from torch.nn import Module
from torch.utils.data import Dataset

import mm_video
from mm_video.trainer.trainer_utils import manual_seed
from mm_video.config import BaseConfig, runner_store
from mm_video.modeling.meter import Meter, DummyMeter
from mm_video.utils.profile import Timer

import logging

logger = logging.getLogger(__name__)

__all__ = ["Runner"]


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

        def is_target(x: Any) -> bool:
            if isinstance(x, dict):
                return "_target_" in x
            if OmegaConf.is_dict(x):
                return "_target_" in x
            return False

        if is_target(dataset_config):
            with Timer("Building dataset from the configuration..."):
                dataset = {split: instantiate(dataset_config, split=split) for split in dataset_splits}
        elif (all(k in ("train", "test", "eval") for k in dataset_config.keys()) and
              all(is_target(v) or v is None for k, v in dataset_config.items())):
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
    def build_meter(meter_config: DictConfig) -> Meter:
        meter = instantiate(meter_config)
        if meter is None:
            logger.info("Meter is not specified.")
            meter = DummyMeter()
        return meter

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

        trainer.train()
        if self.do_eval:
            trainer.evaluate(dataset["eval"])

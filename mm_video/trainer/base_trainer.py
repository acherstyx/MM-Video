# -*- coding: utf-8 -*-
# @Time    : 2022/11/12 22:28
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : base_trainer.py

import os

from tqdm import tqdm
import logging

from hydra.utils import instantiate
from omegaconf import DictConfig
from dataclasses import field, dataclass
from typing import *

import torch
from torch import nn, optim
from torch.utils import data
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist

from mm_video.config.registry import register_trainer_config
from mm_video.modeling.model import ModelBuilderConfig, build_model
from mm_video.modeling.meter import Meter, DummyMeter
from mm_video.data.loader import DataLoaderConfig, build_loader

from mm_video.utils.logging import LogConfig
from mm_video.utils.train_utils import CudaPreFetcher, get_trainable_parameters
from mm_video.utils.checkpoint import save_checkpoint, load_checkpoint, auto_resume, load_model, save_model
from mm_video.utils.writer import get_writer

__all__ = ["BaseTrainer", "BaseTrainerConfig"]

logger = logging.getLogger(__name__)


class BaseTrainer:
    dataloader: Dict[str, data.DataLoader]
    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
    meter: Meter

    def __init__(
            self,
            dataloader, model, optimizer, scheduler, meter,
            test_enable: bool,
            train_enable: bool,
            epoch: int,
            gradient_accumulation_steps: int,
            resume: Optional[str],
            auto_resume: bool,
            clip_norm: Optional[float],
            save_freq: int,
            log_freq: int,
            amp: bool,
            debug: bool,
            write_histogram: bool,
            write_profiler: bool,
            detect_anomaly: bool,
            output_dir: str,
    ):
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.meter = meter

        torch.autograd.set_detect_anomaly(detect_anomaly)

        self.scaler: GradScaler = GradScaler() if amp else None

        self.train_enable = train_enable
        self.test_enable = test_enable

        self.debug = debug
        self.write_profiler = write_profiler
        self.write_histogram = write_histogram
        self.enable_amp = amp

        # initialize/update in `_before_train` method
        self.global_step = self.epoch_start = self.epoch = 0
        self.gradient_accumulation_step = gradient_accumulation_steps

        self.resume = resume
        self.auto_resume = auto_resume
        self.clip_norm = clip_norm

        self.epoch_total = epoch

        self.log_dir = output_dir
        self.log_freq = log_freq
        self.save_freq = save_freq

        self.info()

    def info(self):
        """CUSTOMIZE: to print some information"""
        logger.info("Train Epoch: %d", self.epoch_total)
        trainable_params, all_param = get_trainable_parameters(self.model)
        logger.info("Trainable params: %d", trainable_params)
        logger.info("All params: %d", all_param)
        logger.info("Trainable%%: %f", 100 * trainable_params / all_param)

    def _before_train(self):
        # resume from specified model
        if self.resume is not None:
            logger.info(f"Resume model parameters from {self.resume}.")
            load_model(self.resume, self.model, strict=False)
        # auto resume from checkpoint
        if self.auto_resume:
            logger.info("Auto resume is enabled, recover from the most recent checkpoint.")
            ckpt_dir = os.path.join(self.log_dir, "checkpoint")
            ckpt_file = auto_resume(ckpt_dir)
            if ckpt_file is not None:
                logger.info(f"auto resume from checkpoint: {ckpt_file}")
                # resume from checkpoint
                self.epoch_start = load_checkpoint(ckpt_file, self.model, self.optimizer, self.scheduler,
                                                   restart_train=False)
            else:
                logger.info(f"No checkpoint was found in directory {ckpt_dir}.")
        else:
            logger.debug("Auto resume is disabled.")
        self.global_step = self.epoch_start * (len(self.dataloader["train"]) // self.gradient_accumulation_step) \
            if not self.debug else self.epoch_start * min(len(self.dataloader["train"]), 100)
        self.writer = get_writer(os.path.join(self.log_dir, "tensorboard"), purge_step=self.global_step)

    def _on_train(self):
        for epoch in range(self.epoch_start, self.epoch_total):
            self.epoch = epoch
            logger.debug(f"Epoch {epoch + 1}/{self.epoch_total}")
            torch.cuda.empty_cache()
            if self.train_enable:
                self._before_train_epoch()
                self._on_train_epoch()
                self._after_train_epoch()
            else:
                logger.warning("Training is disabled!")
            torch.cuda.empty_cache()
            if self.test_enable:
                self._before_test_epoch()
                self._on_test_epoch()
                self._after_test_epoch()
            else:
                logger.warning("Testing is disabled!")
            torch.cuda.empty_cache()

    def _after_train(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            save_model(model_file=os.path.join(self.log_dir, "pytorch_model.bin"), model=self.model)

    def _before_train_epoch(self):
        if dist.is_initialized():
            dist.barrier()
        if "train_sampler" in self.dataloader:
            logger.debug(f"set train sampler step to {self.epoch}")
            self.dataloader["train_sampler"].set_epoch(self.epoch)

    def _on_train_epoch(self):
        dataloader = self.dataloader["train"]
        self.model.train()
        if torch.cuda.is_available():  # TODO: add rule based on config
            logger.debug("Building CudaPreFetcher...")
            dataloader = CudaPreFetcher(dataloader)  # prefetch to GPU
            logger.debug("CudaPreFetcher is built successfully.")
        bar = dataloader = tqdm(dataloader,
                                desc=f"Train: {self.epoch + 1}/{self.epoch_total}",
                                dynamic_ncols=True,
                                disable=dist.is_initialized() and dist.get_rank() != 0)
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.log_dir, "profiler")),
            record_shapes=False,
            with_stack=False
        )
        if self.write_profiler:
            prof.start()
        loss_total = 0.
        logger.debug("Running train epoch for-loop...")
        for cur_step, inputs in enumerate(dataloader):
            # forward
            with autocast(enabled=self.enable_amp):
                outputs = self.model(inputs)
                assert "loss" in outputs, \
                    "The model forward function should return a dictionary with the key `loss` during training."
                loss_meta = outputs["loss"]
            # backward
            if isinstance(loss_meta, dict):
                loss = sum([v for _, v in loss_meta.items()])
            else:
                loss = loss_meta
            loss /= self.gradient_accumulation_step
            if not self.enable_amp:
                loss.backward()
            else:
                self.scaler.scale(loss).backward()
            loss_total += loss.detach()
            if self.gradient_accumulation_step > 1:
                bar.set_postfix(
                    {"Accumulation Step": (cur_step + 1) % self.gradient_accumulation_step}
                )
            # write histogram
            with torch.no_grad():
                if self.debug and self.write_histogram and not self.enable_amp:
                    for n, p in self.model.named_parameters():
                        self.writer.add_histogram(f"weight/{n}", p, global_step=self.global_step)
                        if p.grad is not None:
                            self.writer.add_histogram(f"grad/{n}", p.grad, global_step=self.global_step)
            if (cur_step + 1) % self.gradient_accumulation_step == 0:
                # optimize
                if not self.enable_amp:
                    if self.clip_norm is not None:  # clip by norm
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else:
                    if self.clip_norm is not None:  # clip by norm
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()
                # summary
                with torch.no_grad():
                    if cur_step % self.log_freq == 0:
                        if dist.is_initialized():
                            dist.all_reduce(loss)
                            loss = loss / dist.get_world_size()
                            logger.debug(
                                f"loss (rank {dist.get_rank()}, step {self.global_step}): {loss.cpu().detach().numpy()}"
                            )
                        else:
                            logger.debug(f"loss (step {self.global_step}): {loss.cpu().detach().numpy()}")
                        self.writer.add_scalar("train/loss", loss_total, global_step=self.global_step)
                        if isinstance(loss_meta, dict):
                            self.writer.add_scalars("train/loss_meta", loss_meta, global_step=self.global_step)
                        self.writer.add_scalars(
                            "train/lr",
                            {
                                f"param_group_{i}": group["lr"] if self.scheduler is None else group
                                for i, group in enumerate(self.optimizer.param_groups if self.scheduler is None
                                                          else self.scheduler.get_last_lr())
                            },
                            global_step=self.global_step
                        )
                    self.meter.update(
                        inputs=inputs, outputs=outputs,
                        writer=self.writer, main_tag="train", global_step=self.global_step
                    )
                loss_total = 0.
                self.global_step += 1
                if self.write_profiler:
                    prof.step()
            if self.debug and cur_step + 1 >= 100 * self.gradient_accumulation_step:
                logger.warning("Debug mode is enabled, only run for 100 step.")
                break
        logger.debug("Train epoch for-loop finished.")
        if self.write_profiler:
            prof.stop()

    def _after_train_epoch(self):
        # reset optimizer
        self.optimizer.zero_grad()
        # write metric and reset meter
        self.meter.summary(writer=self.writer, main_tag="train", global_step=self.global_step)
        self.meter.reset()
        # save checkpoint
        if (self.epoch + 1) % self.save_freq == 0:
            if not dist.is_initialized() or dist.get_rank() == 0:  # ddp is not enabled or global rank is 0
                save_checkpoint(ckpt_folder=os.path.join(self.log_dir, "checkpoint"),
                                epoch=self.epoch + 1,
                                model=self.model,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                config=None)

    def _before_test_epoch(self):
        pass

    @torch.no_grad()
    def _on_test_epoch(self):
        self.model.eval()
        dataloader = self.dataloader["test"]
        dataloader = tqdm(dataloader, desc=f"Eval epoch {self.epoch + 1}", dynamic_ncols=True,
                          disable=dist.is_initialized() and dist.get_rank() != 0)
        if torch.cuda.is_available():
            dataloader = CudaPreFetcher(dataloader)  # move to GPU
        for inputs in dataloader:
            outputs = self.model(inputs)
            self.meter.update(
                inputs=inputs, outputs=outputs,
                writer=self.writer, main_tag="test", global_step=self.epoch
            )

    def _after_test_epoch(self):
        # write metric and reset meter
        self.meter.summary(writer=self.writer, main_tag="test", global_step=self.epoch)
        self.meter.reset()

    def train(self):
        self._before_train()
        self._on_train()
        self._after_train()

    def eval(self):
        pass


@register_trainer_config(name=f"{BaseTrainer.__qualname__}")
@dataclass
class BaseTrainerConfig:
    _target_: str = f"{__name__}.{BaseTrainer.__qualname__}"

    test_enable: bool = True
    train_enable: bool = True
    epoch: int = 50
    gradient_accumulation_steps: int = 1
    resume: Optional[str] = None
    auto_resume: bool = False
    clip_norm: Optional[float] = None
    save_freq: int = 1
    log_freq: int = 1
    amp: bool = False
    debug: bool = False
    write_histogram: bool = False
    write_profiler: bool = False
    detect_anomaly: bool = False
    output_dir: str = "${log.output_dir}"

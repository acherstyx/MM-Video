# -*- coding: utf-8 -*-
# @Time    : 2022/11/12 22:28
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : base_trainer.py

import os
from abc import ABC, abstractmethod
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

from collections import defaultdict
import contextlib

from mm_video.config.registry import register_trainer_config
from mm_video.modeling.meter import Meter
from mm_video.utils.train_utils import CudaPreFetcher, get_trainable_parameters, compute_total_gradient_norm
from mm_video.utils.checkpoint import save_checkpoint, load_checkpoint, auto_resume, load_model, save_model
from mm_video.utils.writer import get_writer
from mm_video.utils.profile import Timer

__all__ = ["BaseTrainer", "BaseTrainerConfig"]

logger = logging.getLogger(__name__)


class BaseTrainer:
    dataloader: Dict[str, Union[data.DataLoader, data.distributed.DistributedSampler]]
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
            amp: bool,
            debug: bool,
            write_profiler: bool,
            write_loss_and_learning_rate: int,
            write_histogram: Optional[int],
            write_gradient_norm: Optional[int],
            detect_anomaly: bool,
            output_dir: str,
    ):
        assert write_histogram is None or not amp, \
            "If using AMP, `write_histogram_freq` cannot be enabled and must be set to `None`."

        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.meter = meter

        torch.autograd.set_detect_anomaly(detect_anomaly)

        self.train_enable = train_enable
        self.test_enable = test_enable

        self.debug = debug

        self.resume = resume
        self.auto_resume = auto_resume
        self.output_dir = output_dir
        self.save_freq = save_freq

        def get_write_freq(x: Optional[int]):
            assert x is None or type(x) is int
            return float("inf") if x is None else x

        self.write_profiler = write_profiler
        self.write_loss_and_learning_rate_freq = get_write_freq(write_loss_and_learning_rate)
        self.write_histogram_freq = get_write_freq(write_histogram)
        self.write_gradient_norm_freq = get_write_freq(write_gradient_norm)

        self.enable_amp = amp
        self.scaler: GradScaler = GradScaler() if amp else None
        self.clip_norm = clip_norm

        self.epoch_total = epoch
        self.global_step = self.epoch_start = self.epoch = 0
        self.gradient_accumulation_step = gradient_accumulation_steps

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
            ckpt_dir = os.path.join(self.output_dir, "checkpoint")
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
        self.writer = get_writer(os.path.join(self.output_dir, "tensorboard"), purge_step=self.global_step)

    def _on_train(self):
        for epoch in range(self.epoch_start, self.epoch_total):
            self.epoch = epoch
            logger.debug(f"Epoch {epoch + 1}/{self.epoch_total}")
            if self.train_enable:
                self._before_train_epoch()
                self._on_train_epoch()
                self._after_train_epoch()
            else:
                logger.warning("Training is disabled!")
            if self.test_enable:
                self._before_test_epoch()
                self._on_test_epoch()
                self._after_test_epoch()
            else:
                logger.warning("Testing is disabled!")

    def _after_train(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            save_model(model_file=os.path.join(self.output_dir, "pytorch_model.bin"), model=self.model)

    def _before_train_epoch(self):
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()
        if "train_sampler" in self.dataloader:
            logger.debug(f"set train sampler step to {self.epoch}")
            self.dataloader["train_sampler"].set_epoch(self.epoch)

    def _on_train_epoch(self):
        dataloader = self.dataloader["train"]
        if torch.cuda.is_available():
            logger.debug("Building CudaPreFetcher...")
            dataloader = CudaPreFetcher(dataloader)  # prefetch to GPU
            logger.debug("CudaPreFetcher is built successfully.")
        progress_bar = tqdm(desc=f"Train: {self.epoch + 1}/{self.epoch_total}",
                            dynamic_ncols=True,
                            total=len(dataloader) // self.gradient_accumulation_step,
                            disable=dist.is_initialized() and dist.get_rank() != 0)
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.output_dir, "profiler")),
            record_shapes=False,
            with_stack=False
        )
        if self.write_profiler:
            prof.start()

        loss_total = 0.0
        loss_meta_total = defaultdict(float)

        logger.debug("Running train epoch for-loop...")
        for cur_step, inputs in enumerate(dataloader):
            outputs, loss, loss_meta = self._train_step(inputs=inputs)

            loss_total += loss
            for k, v in loss_meta.items():
                loss_meta_total[k] += v

            if self.gradient_accumulation_step > 1:
                progress_bar.set_postfix_str(f"Accumulation Step={(cur_step + 1) % self.gradient_accumulation_step}")
            if (cur_step + 1) % self.gradient_accumulation_step == 0:
                # summary
                with torch.no_grad():
                    if (cur_step + 1) % self.write_histogram_freq == 0:
                        self._write_histogram()
                    if (cur_step + 1) % self.write_gradient_norm_freq == 0:
                        self._write_total_gradient_norm()
                    if (cur_step + 1) % self.write_loss_and_learning_rate_freq == 0:
                        # noinspection PyTypeChecker
                        self._write_loss_and_learning_rate(loss=loss_total, loss_meta=loss_meta_total)
                    self.meter.update(
                        inputs=inputs, outputs=outputs,
                        writer=self.writer, main_tag="train", global_step=self.global_step
                    )
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
                self.global_step += 1
                loss_total = 0.
                loss_meta_total = defaultdict(float)
                progress_bar.update()
                if self.write_profiler:
                    prof.step()
            if self.debug and cur_step + 1 >= 100 * self.gradient_accumulation_step:
                logger.warning("Debug mode is enabled, only run for 100 step.")
                break
        logger.debug("Train epoch for-loop finished.")
        if self.write_profiler:
            prof.stop()

    def _train_step(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        """
        Performs a single training step on the model.
        """
        self.model.train()

        if self.enable_amp:
            autocast_context_manager = autocast()
        else:
            autocast_context_manager = contextlib.nullcontext()

        # Forward pass
        with autocast_context_manager:
            outputs = self.model(inputs)
            assert "loss" in outputs, \
                "The model forward function should return a dictionary with the key `loss` during training."
            loss = outputs["loss"]

        # Sum up loss if returned loss is a dictionary
        if isinstance(loss, dict):
            loss_meta = {k: v for k, v in loss.items()}
            loss = sum([v for _, v in loss.items()])
        else:
            loss_meta = {}

        if self.enable_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Divide with gradient accumulation steps and return
        loss = loss.detach() / self.gradient_accumulation_step
        loss_meta = {k: v.detach() / self.gradient_accumulation_step for k, v in loss_meta.items()}
        return outputs, loss, loss_meta

    def _after_train_epoch(self):
        # reset optimizer
        self.optimizer.zero_grad()
        # write metric and reset meter
        self.meter.summary(writer=self.writer, main_tag="train", global_step=self.global_step)
        self.meter.reset()
        # save checkpoint
        if (self.epoch + 1) % self.save_freq == 0:
            if not dist.is_initialized() or dist.get_rank() == 0:  # ddp is not enabled or global rank is 0
                save_checkpoint(ckpt_folder=os.path.join(self.output_dir, "checkpoint"),
                                epoch=self.epoch + 1,
                                model=self.model,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                config=None)
        torch.cuda.empty_cache()

    def _before_test_epoch(self):
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()
        if "test_sampler" in self.dataloader:
            logger.debug(f"set test sampler step to {self.epoch}")
            self.dataloader["test_sampler"].set_epoch(self.epoch)

    @torch.no_grad()
    def _on_test_epoch(self):
        self.model.eval()
        dataloader = self.dataloader["test"]
        progress_bar = tqdm(desc=f"Eval epoch {self.epoch + 1}", dynamic_ncols=True, total=len(dataloader),
                            disable=dist.is_initialized() and dist.get_rank() != 0)
        if torch.cuda.is_available():
            dataloader = CudaPreFetcher(dataloader)  # move to GPU
        for inputs in dataloader:
            outputs = self.model(inputs)
            self.meter.update(
                inputs=inputs, outputs=outputs,
                writer=self.writer, main_tag="test", global_step=self.epoch
            )
            progress_bar.update()

    def _after_test_epoch(self):
        # write metric and reset meter
        self.meter.summary(writer=self.writer, main_tag="test", global_step=self.epoch)
        self.meter.reset()
        torch.cuda.empty_cache()

    def train(self):
        self._before_train()
        self._on_train()
        self._after_train()

    def eval(self):
        pass

    @torch.no_grad()
    def _write_histogram(self):
        """
        Writes histograms of model parameters and gradients.

        The histograms are written to TensorBoard under the tags:
        "weights/{parameter name}" and "grads/{parameter name}", respectively.
        """
        with Timer("Writing histogram..."):
            for n, p in self.model.named_parameters():
                self.writer.add_histogram(f"weight/{n}", p.detach().float(), global_step=self.global_step)
                if p.grad is not None:
                    self.writer.add_histogram(f"grad/{n}", p.grad.detach().float(), global_step=self.global_step)

    @torch.no_grad()
    def _write_total_gradient_norm(self):
        """
        Compute and writes total gradient norm.

        This function should be called before gradient clipping. The total gradient norm over all model parameters is
        computed, and written to TensorBoard under the tag "train/norm".
        """
        with Timer("Writing total gradient norm..."):
            total_norm = compute_total_gradient_norm(self.model)
            self.writer.add_scalar("train/norm", total_norm, global_step=self.global_step)

    @torch.no_grad()
    def _write_loss_and_learning_rate(
            self, loss: torch.Tensor, loss_meta: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ):
        """
        Writes training loss, loss metadata, and learning rate to TensorBoard.

        :param loss: Training loss.
        :param loss_meta:
        :return:
        """
        if dist.is_initialized():
            dist.all_reduce(loss)
            loss /= dist.get_world_size()
            for k in sorted(loss_meta.keys()):
                dist.all_reduce(loss_meta[k])
                loss_meta[k] /= dist.get_world_size()
        logger.debug(f"Step: {self.global_step} | Loss: {loss.cpu().detach().numpy()}")
        self.writer.add_scalar("train/loss", loss, global_step=self.global_step)
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


@register_trainer_config(name=f"{BaseTrainer.__qualname__}")
@dataclass
class BaseTrainerConfig:
    _target_: str = f"{__name__}.{BaseTrainer.__qualname__}"

    # Enable/Disable train/test loop
    train_enable: bool = True
    test_enable: bool = True

    epoch: int = 5
    amp: bool = False
    gradient_accumulation_steps: int = 1
    detect_anomaly: bool = False
    clip_norm: Optional[float] = None

    # Resume model from pretrained parameters
    resume: Optional[str] = None
    # Resume from checkpoint saved during training, including the status of optimizer, scheduler, etc.
    auto_resume: bool = False

    # Enable/disable PyTorch profiler
    write_profiler: bool = False
    # How often to write training loss, loss metadata, and learning rate to TensorBoard.
    # Set to `None` to disable.
    write_loss_and_learning_rate: Optional[int] = 1
    # How often to write parameter histograms to TensorBoard. Set to `None` to disable.
    write_histogram: Optional[int] = None
    # How often to compute and write total gradient norm to TensorBoard. Set to `None` to disable.
    write_gradient_norm: Optional[int] = None

    output_dir: str = "${hydra:runtime.output_dir}"
    save_freq: int = 1

    debug: bool = False

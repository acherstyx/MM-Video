# -*- coding: utf-8 -*-
# @Time    : 2022/11/12 22:28
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : trainer.py
import glob
import math
import os
import re

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging

from hydra.utils import get_object
from dataclasses import dataclass, asdict, field
import json
from typing import *

import torch
from torch import nn, optim
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    CPUOffload
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy
)

from collections import defaultdict
import contextlib
from functools import partial

from mm_video.config import trainer_store
from mm_video.modeling.meter import Meter
from mm_video.modeling.optimization import get_linear_schedule_with_warmup
from mm_video.utils.train_utils import (
    CudaPreFetcher, get_trainable_parameters, compute_total_gradient_norm,
    save_rng_state, load_rng_state
)
from mm_video.utils.distributed import get_world_size, get_local_rank, get_master_addr, get_master_port, get_rank
from mm_video.utils.writer import get_writer
from mm_video.utils.profile import Timer
from .trainer_utils import barrier, get_module_class_from_name, load_state_dict, unwrap_model, get_write_freq, \
    has_length
from .training_configs import TrainingConfig, DebugConfig, TrainingStrategyConfig, TrainingStrategy, DataLoaderConfig

__all__ = [
    "Trainer", "TrainerState"
]

logger = logging.getLogger(__name__)

PREFIX_CHECKPOINT_DIR = "checkpoint"

MODEL_NAME = "pytorch_model"
OPTIMIZER_NAME = "optimizer"
SCHEDULER_NAME = "scheduler"
TRAINER_STATE_NAME = "trainer_state"
MODEL_NAME_BIN = f"{MODEL_NAME}.bin"
OPTIMIZER_NAME_BIN = f"{OPTIMIZER_NAME}.bin"
SCHEDULER_NAME_BIN = f"{SCHEDULER_NAME}.bin"
TRAINER_STATE_NAME_BIN = f"{TRAINER_STATE_NAME}.json"


@dataclass
class TrainerState:
    # Trained epochs and steps
    epoch: float = 0
    global_step: int = 0

    save_epochs: Optional[int] = None
    save_steps: Optional[int] = None
    eval_steps: Optional[int] = None
    eval_epochs: Optional[int] = None
    logging_steps: Optional[int] = None

    log_history: List[Dict[str, float]] = field(default_factory=list)

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """Create an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))

    def should_evaluate_step(self, global_step: Optional[int] = None) -> bool:
        global_step = self.global_step if global_step is None else global_step
        return global_step % get_write_freq(self.eval_steps) == 0

    def should_evaluate_epoch(self, epoch: Optional[float] = None) -> bool:
        epoch = self.epoch if epoch is None else epoch
        epoch = float(epoch)
        return epoch % get_write_freq(self.eval_epochs) == 0

    @property
    def should_evaluate(self) -> bool:
        if self.global_step % get_write_freq(self.eval_steps) == 0:
            return True
        if self.epoch % get_write_freq(self.eval_epochs) == 0:
            return True
        return False

    @property
    def should_save(self) -> bool:
        if self.global_step % get_write_freq(self.save_steps) == 0:
            return True
        if self.epoch % get_write_freq(self.save_epochs) == 0:
            return True
        return False

    @property
    def should_log(self) -> bool:
        return self.global_step % get_write_freq(self.logging_steps) == 0


# Set `zen_partial=True` if you inherit from this trainer
@trainer_store(zen_partial=True)
class Trainer:
    def __init__(
            self,
            model: nn.Module, meter: Meter,
            train_dataset: Optional[Dataset] = None, eval_dataset: Optional[Dataset] = None,
            optimizer: Optional[optim.Optimizer] = None, scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
            # Configs
            output_dir: str = "${hydra:runtime.output_dir}",
            training: TrainingConfig = TrainingConfig(),
            dataloader: DataLoaderConfig = DataLoaderConfig(),
            training_strategy: TrainingStrategyConfig = TrainingStrategyConfig(),
            debug: DebugConfig = DebugConfig()
    ):
        self.output_dir = output_dir
        self.training_cfg: TrainingConfig = training
        self.dataloader_cfg: DataLoaderConfig = dataloader
        self.training_strategy_cfg: TrainingStrategyConfig = training_strategy
        self.debug_cfg = debug

        assert self.training_cfg.write_histogram is None or not self.training_cfg.amp, \
            "If using AMP, `write_histogram_freq` cannot be enabled and must be set to `None`."
        assert self.training_cfg.gradient_accumulation_steps >= 1, \
            "`gradient_accumulation_steps` must be an integer greater or equal to 1."

        torch.autograd.set_detect_anomaly(self.training_cfg.detect_anomaly)
        # Initialize distribution
        if not dist.is_initialized():
            if get_local_rank() == 0:
                logger.info("Distributed training config:\n\tMaster IP: %s\n\tMaster port: %s\n\tWorld size: %s",
                            get_master_addr(), get_master_port(), get_world_size())
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))  # get RANK from environment

        self.dataset = None
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.model_wrapped = self.model = model

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.meter: Meter = meter  # TODO: Support a list of meters
        self.scaler: GradScaler = GradScaler() if self.training_cfg.amp else None

        self.state = TrainerState(
            save_epochs=self.training_cfg.save_epochs,
            save_steps=self.training_cfg.save_steps,
            eval_steps=self.training_cfg.eval_steps,
            eval_epochs=self.training_cfg.eval_epochs,
            logging_steps=self.training_cfg.logging_steps
        )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

    def create_optimizer(self):
        logger.debug("Creating optimizer...")
        self.optimizer = optim.AdamW(self.model_wrapped.parameters(), lr=self.training_cfg.learning_rate)
        logger.debug("Optimizer created successfully.")

    def create_scheduler(self, num_training_steps: int, optimizer: optim.Optimizer):
        logger.debug("Creating scheduler...")
        warmup_steps = 0
        assert (self.training_cfg.warmup_ratio is None) or (self.training_cfg.warmup_steps is None), \
            "Both warmup_ratio and warmup_steps should not be set simultaneously."
        if self.training_cfg.warmup_ratio is not None:
            warmup_steps = int(num_training_steps * self.training_cfg.warmup_ratio)
        if self.training_cfg.warmup_steps is not None:
            warmup_steps = self.training_cfg.warmup_steps

        self.scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                         num_warmup_steps=warmup_steps,
                                                         num_training_steps=num_training_steps)
        logger.debug("Scheduler created successfully. Warmup steps: %d", warmup_steps)

    def get_train_dataloader(self) -> Tuple[DataLoader, Optional[Sampler]]:
        if self.train_dataset is None:
            raise ValueError("Training requires a train_dataset.")

        cfg = self.dataloader_cfg
        train_dataset = self.train_dataset

        if get_world_size() > 1:
            sampler = data.distributed.DistributedSampler(train_dataset, shuffle=cfg.shuffle)
        elif cfg.shuffle:
            sampler = RandomSampler(train_dataset)
        else:
            sampler = None
        dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.train_batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            collate_fn=get_object(cfg.collate_fn) if cfg.collate_fn is not None else None,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last,
            prefetch_factor=cfg.prefetch_factor,
            persistent_workers=cfg.persistent_workers
        )
        return dataloader, sampler

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Evaluation requires an eval_dataset.")

        cfg = self.dataloader_cfg
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        return DataLoader(
            eval_dataset,
            batch_size=cfg.eval_batch_size,
            collate_fn=get_object(cfg.collate_fn) if cfg.collate_fn is not None else None,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers
        )

    @property
    def should_write(self) -> bool:
        return get_rank() == 0

    def save_model(self, output_dir: Optional[str] = None, state_dict: Optional[Any] = None):
        """
        Save model (state dict) to disk. Make sure this function is only executed on process 0.

        @param output_dir:
        @param state_dict:
        """
        output_dir = output_dir if output_dir is not None else self.output_dir
        model_file = os.path.join(output_dir, MODEL_NAME_BIN)
        logger.info(f"Saving model to {model_file}")

        if state_dict is None:
            # Call the state_dict on each rank before saving on rank 0, required by FSDP model
            model = unwrap_model(self.model_wrapped)
            state_dict = model.state_dict()

        if self.should_write:
            os.makedirs(output_dir, exist_ok=True)
            torch.save(state_dict, model_file)

    def _save_checkpoint(self, model: nn.Module = None):
        """
        Save current training status to checkpoint

        """
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        output_dir = os.path.join(self.output_dir, checkpoint_folder)
        logger.debug("Saving trainer checkpoint to %s", output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Save model status
        model = model if model is not None else self.model_wrapped
        state_dict = model.state_dict()
        # Save rng status
        save_rng_state(output_dir)
        if self.should_write:
            self.save_model(output_dir=output_dir, state_dict=state_dict)
        # Save optimizer
        if self.should_write:
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME_BIN))
        # Save scheduler
        if self.should_write and self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME_BIN))
        # Save trainer state
        if self.should_write:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME_BIN))

        barrier()
        logger.info("Checkpoint is saved to %s", output_dir)

    def _load_from_checkpoint(self, checkpoint: str, model=None):
        if model is None:
            model = self.model

        """Load model states"""
        model_state_dict = torch.load(os.path.join(checkpoint, MODEL_NAME_BIN), map_location="cpu")
        model.load_state_dict(model_state_dict)

    def _load_optimizer_and_scheduler(self, checkpoint):
        # Load optimizer
        optimizer_file = os.path.join(checkpoint, OPTIMIZER_NAME_BIN)
        if os.path.exists(optimizer_file):
            self.optimizer.load_state_dict(torch.load(optimizer_file, map_location="cpu"))
        # Load scheduler
        scheduler_file = os.path.join(checkpoint, SCHEDULER_NAME_BIN)
        if os.path.exists(scheduler_file):
            self.scheduler.load_state_dict(torch.load(scheduler_file, map_location="cpu"))

    def _get_last_checkpoint(self) -> Optional[str]:
        """
        Get the last checkpoint listed in output directory

        """
        checkpoint_folder_pattern = f"{PREFIX_CHECKPOINT_DIR}-[0-9]*"
        checkpoint_folders = glob.glob(os.path.join(self.output_dir, checkpoint_folder_pattern))
        checkpoint_folders = [f for f in checkpoint_folders if os.path.isdir(f)]
        checkpoint_folders = [f for f in checkpoint_folders if os.path.exists(os.path.join(f, TRAINER_STATE_NAME_BIN))]
        logger.debug("checkpoints found in the output directory: %s", checkpoint_folders)
        if checkpoint_folders:
            last_checkpoint_folder = sorted(checkpoint_folders, key=lambda x: int(re.findall(r"\d+", x)[-1]))[-1]
            logger.debug("Found the last checkpoint at: %s.", last_checkpoint_folder)
            return last_checkpoint_folder
        else:
            logger.debug("No previous checkpoint found in the output directory.")

    @staticmethod
    def build_loader(datasets: Dict[str, data.Dataset], cfg: DataLoaderConfig):
        assert all(split in ("train", "test", "eval") for split in datasets.keys()), \
            f"Invalid split found in {datasets.keys()}. Must be one of 'train', 'test', or 'eval'."
        timer = Timer(msg="Building dataloader...")
        world_size = get_world_size()
        loader_and_sampler = {}
        for split, dataset in datasets.items():
            shuffle = cfg.shuffle if split == "train" else False
            batch_size = getattr(cfg, f"{split}_batch_size")
            if batch_size is None:
                batch_size = getattr(cfg, "batch_size")
            collate_fn = get_object(cfg.collate_fn) if cfg.collate_fn is not None else None

            sampler = data.distributed.DistributedSampler(dataset, shuffle=shuffle) if world_size > 1 else None

            loader = data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False if world_size > 1 else shuffle,
                sampler=sampler,
                num_workers=cfg.num_workers,
                collate_fn=collate_fn,
                pin_memory=cfg.pin_memory,
                persistent_workers=False,
                prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
                multiprocessing_context=cfg.multiprocessing_context if cfg.num_workers else None
            )

            loader_and_sampler[split] = loader
            if sampler is not None:
                loader_and_sampler[f"{split}_sampler"] = sampler
        timer.end()
        return loader_and_sampler

    def _wrap_model(self, model: nn.Module, training: bool = True):
        cfg = self.training_strategy_cfg

        # Move to GPU
        if cfg.strategy in (TrainingStrategy.ddp, TrainingStrategy.fsdp):
            logger.debug("Moving model to device: %s...", torch.cuda.current_device())
            model.cuda()

        # Do not wrap model if not training.
        if not training:
            logger.debug("Not training, return unwrapped model.")
            return model

        # Avoid wrap model more than once.
        if isinstance(model, (FullyShardedDataParallel, nn.DataParallel)):
            return model

        # wrap model
        logger.debug("Applying training strategy...")
        if cfg.strategy in (TrainingStrategy.ddp, TrainingStrategy.fsdp):
            logger.debug("Model is moved to device: %s", torch.cuda.current_device())
            if cfg.strategy == TrainingStrategy.ddp:
                logger.debug("Building DistributedDataParallel, check whether the program is hanging...")
                model = nn.parallel.DistributedDataParallel(
                    model,
                    find_unused_parameters=cfg.ddp_find_unused_parameters
                )
            elif cfg.strategy == TrainingStrategy.fsdp:
                logger.debug("Building FullyShardedDataParallel, check whether the program is hanging...")

                # From Hugging Face Trainer
                auto_wrap_policy = None
                if cfg.fsdp_transformer_layer_cls_to_wrap is not None:
                    transformer_cls_to_wrap = set()
                    for layer_class in cfg.fsdp_transformer_layer_cls_to_wrap:
                        transformer_cls = get_module_class_from_name(model, layer_class)
                        if transformer_cls is None:
                            raise Exception("Could not find the transformer layer class to wrap in the model.")
                        else:
                            transformer_cls_to_wrap.add(transformer_cls)
                    auto_wrap_policy = partial(
                        transformer_auto_wrap_policy,
                        transformer_layer_cls=transformer_cls_to_wrap
                    )

                self.model = model = FullyShardedDataParallel(
                    model,
                    cpu_offload=CPUOffload(offload_params=cfg.fsdp_offload),
                    auto_wrap_policy=auto_wrap_policy,
                    sync_module_states=cfg.fsdp_sync_module_states,
                    use_orig_params=cfg.fsdp_use_orig_params,
                    sharding_strategy=cfg.fsdp_sharding_strategy
                )
            else:
                raise RuntimeError(f"Training strategy '{cfg.strategy}' is not supported!")
        elif cfg.strategy == TrainingStrategy.cpu:
            pass
        else:
            raise RuntimeError(f"Training strategy '{cfg.strategy}' is not supported!")
        logger.debug("Model is wrapped successfully.")
        return model

    def _prefetch_to_gpu(self, dataloader: data.DataLoader) -> data.DataLoader:
        """
        Wraps the given dataloader with `CudaPreFetcher` to prefetch tensors to GPU.

        This transformation is only applied when the training strategy is either ddp or FSDP.
        Otherwise, the dataloader remains unchanged.

        """
        if self.training_strategy_cfg.strategy in (TrainingStrategy.ddp, TrainingStrategy.fsdp):
            logger.debug("Building CudaPreFetcher. "
                         "This might take a moment as it waits for all Torch DataLoader workers to initialize...")
            dataloader = CudaPreFetcher(dataloader)
            logger.debug("CudaPreFetcher successfully built.")
        return dataloader

    def training_step(
            self,
            model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Performs a single training step on the model.
        """
        model.train()
        if self.training_cfg.amp:
            autocast_context_manager = autocast()
        else:
            autocast_context_manager = contextlib.nullcontext()

        # Forward pass
        with autocast_context_manager:
            outputs = model(inputs)
            assert "loss" in outputs, \
                "The model forward function should return a dictionary with the key `loss` during training."
            loss = outputs["loss"]

        # Sum up loss if returned loss is a dictionary
        if isinstance(loss, dict):
            loss_meta = {k: v for k, v in loss.items()}
            loss = sum([v for _, v in loss.items()])
        else:
            loss_meta = {}

        if self.training_cfg.amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Divide with gradient accumulation steps and return
        loss = loss.detach() / self.training_cfg.gradient_accumulation_steps
        loss_meta = {k: v.detach() / self.training_cfg.gradient_accumulation_steps for k, v in loss_meta.items()}
        return outputs, loss, loss_meta

    def train(self):
        if type(self.training_cfg.resume_from_checkpoint) is str:
            resume_from_checkpoint = self.training_cfg.resume_from_checkpoint
            assert os.path.isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME_BIN)), \
                (f"Invalid checkpoint: {TRAINER_STATE_NAME_BIN} is not found in the checkpoint folder: "
                 f"{resume_from_checkpoint}")
        elif self.training_cfg.resume_from_checkpoint:
            # Load from last checkpoint
            resume_from_checkpoint = self._get_last_checkpoint()
        else:
            resume_from_checkpoint = None

        self.training_loop(resume_from_checkpoint=resume_from_checkpoint)

    def training_loop(self, resume_from_checkpoint: Optional[str] = None):
        torch.cuda.empty_cache()

        gradient_accumulation_steps = self.training_cfg.gradient_accumulation_steps

        train_dataloader, train_sampler = self.get_train_dataloader()

        per_device_train_batch_size = self.dataloader_cfg.train_batch_size
        total_train_batch_size = per_device_train_batch_size * gradient_accumulation_steps * get_world_size()

        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_train_epochs = self.training_cfg.num_train_epochs
            # TODO: Handle other cases where the dataloader is not torch.utils.data.Dataloader
            num_examples = len(train_dataloader.dataset)
            max_steps = math.ceil(self.training_cfg.num_train_epochs * num_update_steps_per_epoch)
        else:
            # TODO: Support 'max_steps' option
            raise NotImplementedError("Dataset do not have a length, not supported.")

        trainable_params, all_param, trainable_params_names = get_trainable_parameters(self.model)

        # Wrap model before training
        model = self._wrap_model(self.model)
        if self.training_strategy_cfg.strategy == TrainingStrategy.FSDP:
            self.model = self.model_wrapped = model
        if model is not self.model:
            self.model_wrapped = model

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Resume from specified model, use for load pretrained weight
        if self.training_cfg.resume is not None:
            logger.info(f"Resume model parameters from {self.training_cfg.resume}.")
            load_state_dict(self.model_wrapped, model_file=self.training_cfg.resume, strict=False)

        writer = get_writer(os.path.join(self.output_dir, "tensorboard"), purge_step=self.state.global_step)

        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Continuing training from a checkpoint
        if resume_from_checkpoint is not None:
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME_BIN))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            steps_trained_in_current_epoch = self.state.global_step % num_update_steps_per_epoch

            logger.info("Continuing training from checkpoint: %s", resume_from_checkpoint)
            logger.info("Continuing training from epoch %d", epochs_trained)
            logger.info("Continuing training from global step %d", self.state.global_step)

        # Load from checkpoint
        if resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model=self.model_wrapped)
            self._load_optimizer_and_scheduler(resume_from_checkpoint)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {per_device_train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {self.training_cfg.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of all parameters = {all_param:,}")
        logger.info(f"  Number of trainable parameters = {trainable_params:,}")
        logger.info(f"  Trainable percentage = {100 * trainable_params / all_param:,}")
        logger.debug("Full list of parameters (* indicates frozen parameters):\n\t%s",
                     '\n\t'.join([(p if p in trainable_params_names else f"{p} *")
                                  for p, _ in self.model.named_parameters()]))
        logger.info("****************************")

        self.optimizer.zero_grad()
        model.zero_grad()

        # Skip the fist epochs_trained epochs to get the random state of the dataloader at the right point
        skip_global_step = 0
        for epoch in range(epochs_trained):
            iter(train_dataloader)
            for step in range(len(train_dataloader)):
                if step % gradient_accumulation_steps == 0:
                    skip_global_step += 1
                    if self.state.should_evaluate_step(skip_global_step):
                        iter(self.get_eval_dataloader())
            if self.state.should_evaluate_epoch(epoch):
                iter(self.get_eval_dataloader())

        for epoch in range(epochs_trained, num_train_epochs):
            logger.debug(f"Train Epoch {epoch + 1}/{self.training_cfg.num_train_epochs}")

            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
                logger.debug(f"Set train sampler step to {epoch}")

            steps_in_epoch = (
                len(train_dataloader) if len_dataloader is not None
                else max_steps * gradient_accumulation_steps
            )

            train_iterator = self._prefetch_to_gpu(train_dataloader)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                load_rng_state(resume_from_checkpoint)

            prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=5, warmup=5, active=5, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    os.path.join(self.output_dir, "profiler")),
                record_shapes=False,
                with_stack=False
            ) if self.training_cfg.write_profiler else None
            if self.training_cfg.write_profiler:
                logger.warning("Torch profiler is enabled, performance may be impacted.")
                prof.start()

            progress_bar = tqdm(
                desc=f"Train: {epoch + 1}/{num_train_epochs}",
                dynamic_ncols=True,
                total=steps_in_epoch // gradient_accumulation_steps,
                disable=get_local_rank() != 0  # Only display on local rank 0
            )

            loss_total = 0.0
            loss_meta_total = defaultdict(float)

            for step, inputs in enumerate(train_iterator):
                # Save model inputs
                if self.debug_cfg.enable and self.debug_cfg.save_inputs:
                    if self.debug_cfg.save_inputs_for_each_step:
                        inputs_save_path = os.path.join(self.output_dir, "inputs",
                                                        f"rank_{dist.get_rank()}_step_{self.state.global_step}.pt")
                    else:
                        inputs_save_path = os.path.join(self.output_dir, "inputs", f"rank_{dist.get_rank()}.pt")
                    os.makedirs(os.path.dirname(inputs_save_path), exist_ok=True)
                    torch.save(inputs, inputs_save_path)

                # Skip already trained steps if resuming from checkpoint
                if steps_trained_in_current_epoch > 0:
                    if step % gradient_accumulation_steps == 0:
                        progress_bar.update()  # TODO: Use a dedicate progress bar for skipping
                        steps_trained_in_current_epoch -= 1
                        skip_global_step += 1
                        if self.state.should_evaluate_step(skip_global_step):
                            iter(self.get_eval_dataloader())
                    if steps_trained_in_current_epoch == 0:
                        logger.debug("Resuming random status from checkpoint at: %s", resume_from_checkpoint)
                        load_rng_state(resume_from_checkpoint)
                        if skip_global_step != self.state.global_step:
                            logger.warning("The skipped training step does not match the global step logged in the "
                                           "checkpoint, resume may fail.")
                    # Skip this step
                    continue

                outputs, loss, loss_meta = self.training_step(model=model, inputs=inputs)

                loss_total += loss
                for k, v in loss_meta.items():
                    loss_meta_total[k] += v

                is_last_step_and_steps_less_than_grad_acc = (
                        gradient_accumulation_steps >= steps_in_epoch == (step + 1)
                )

                if self.training_cfg.gradient_accumulation_steps > 1:
                    progress_bar.set_postfix({
                        "Accumulation Steps": (step + 1) % self.training_cfg.gradient_accumulation_steps
                    })
                if (
                        (step + 1) % gradient_accumulation_steps == 0
                        or
                        is_last_step_and_steps_less_than_grad_acc
                ):
                    # Summary
                    with torch.no_grad():
                        if (step + 1) % get_write_freq(self.training_cfg.write_histogram) == 0:
                            self._write_histogram()
                        if (step + 1) % get_write_freq(self.training_cfg.write_gradient_norm) == 0:
                            self._write_total_gradient_norm(writer)

                    # Clip by norm
                    if self.training_cfg.clip_norm is not None:
                        if self.training_cfg.amp:
                            self.scaler.unscale_(self.optimizer)

                        if self.training_strategy_cfg.strategy == TrainingStrategy.fsdp:
                            model.clip_grad_norm_(self.training_cfg.clip_norm)
                        else:
                            nn.utils.clip_grad_norm_(model.parameters(), self.training_cfg.clip_norm)

                    # Optimize
                    if self.training_cfg.amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.model.zero_grad()
                    if self.scheduler is not None:
                        self.scheduler.step()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch

                    if self.training_cfg.write_profiler:
                        prof.step()

                    self._maybe_log_save_evaluate(inputs, outputs, writer, loss_total, loss_meta_total, model,
                                                  epoch_end=False)

                    loss_total = 0.
                    loss_meta_total = defaultdict(float)
                    progress_bar.update()
                # Exit if debug
                if (self.debug_cfg.enable and
                        step + 1 >= self.debug_cfg.max_train_steps * self.training_cfg.gradient_accumulation_steps):
                    logger.warning("Debug mode is enabled, only run for %s step.",
                                   self.debug_cfg.max_train_steps)
                    break
            progress_bar.close()
            logger.debug("Train epoch for-loop finished.")

            if self.training_cfg.write_profiler:
                prof.stop()

            # reset optimizer
            self.model.zero_grad()
            self.optimizer.zero_grad()
            # write metric and reset meter
            self.meter.summary(writer=writer, main_tag="train", global_step=self.state.global_step)
            self.meter.reset()
            # save checkpoint
            self._maybe_log_save_evaluate(None, None, writer, loss_total, loss_meta_total, model, epoch_end=True)

            if self.state.should_evaluate_epoch():
                self.evaluate()

        state_dict = self.model_wrapped.state_dict()
        if self.should_write:
            self.save_model(state_dict=state_dict)

    def _maybe_log_save_evaluate(self, inputs, outputs, writer, loss_total, loss_meta_total, model, epoch_end):
        """Call in training loop"""
        if not epoch_end and self.state.should_log:
            self.log({"loss": loss_total.item()})
            self._write_loss_and_learning_rate(writer, loss_total, loss_meta_total)

        # Update meter after each train step
        if not epoch_end:
            self.meter.update(
                inputs=inputs, outputs=outputs,
                writer=writer, main_tag="train", global_step=self.state.global_step
            )

        if self.state.should_evaluate:
            self.evaluate()

        if self.state.should_save:
            self._save_checkpoint(model)

    def prediction_step(self, model, inputs):
        with torch.no_grad():
            outputs = model(inputs)
        return outputs

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str = "",
            metric_key_prefix: str = ""
    ) -> Dict[str, float]:
        torch.cuda.empty_cache()

        writer = get_writer(os.path.join(self.output_dir, "tensorboard"), purge_step=self.state.global_step)

        model = self._wrap_model(self.model, training=False)
        model.eval()

        dataloader = self._prefetch_to_gpu(dataloader)

        progress_bar = tqdm(
            desc=description,
            dynamic_ncols=True,
            total=len(dataloader),
            disable=not self.should_write
        )
        for step, inputs in enumerate(dataloader):
            outputs = self.prediction_step(model, inputs)
            self.meter.update(
                inputs=inputs,
                outputs=outputs,
                writer=writer,
                main_tag="test",
                global_step=int(self.state.epoch)
            )
            progress_bar.update()
            # Exit if debug
            if self.debug_cfg.enable and step + 1 >= self.debug_cfg.max_test_steps:
                logger.warning("Debug mode is enabled, only run for %s step.", self.debug_cfg.max_test_steps)
                break
        # write metric and reset meter
        metrics = self.meter.summary(writer=writer, main_tag="test", global_step=int(self.state.epoch))
        # Replace None return with an empty dict
        metrics = {} if metrics is None else metrics
        self.meter.reset()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return metrics

    @torch.no_grad()
    def evaluate(
            self,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            metric_key_prefix: str = ""
    ) -> Dict[str, float]:
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        metrics = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            metric_key_prefix=metric_key_prefix
        )
        return metrics

    def log(self, logs: Dict[str, float]):
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)

    @torch.no_grad()
    def _write_histogram(self, writer: SummaryWriter):
        """
        Writes histograms of model parameters and gradients.

        The histograms are written to TensorBoard under the tags:
        "weights/{parameter name}" and "grads/{parameter name}", respectively.
        """
        with Timer("Writing histogram..."):
            for n, p in self.model.named_parameters():
                writer.add_histogram(f"weight/{n}", p.detach().float(), global_step=self.state.global_step)
                if p.grad is not None:
                    writer.add_histogram(f"grad/{n}", p.grad.detach().float(), global_step=self.state.global_step)

    @torch.no_grad()
    def _write_total_gradient_norm(self, writer: SummaryWriter):
        """
        Compute and writes total gradient norm.

        This function should be called before gradient clipping. The total gradient norm over all model parameters is
        computed, and written to TensorBoard under the tag "train/norm".
        """
        with Timer("Writing total gradient norm..."):
            total_norm = compute_total_gradient_norm(self.model)
            logger.debug(f"Step: {self.state.global_step} | Total gradient norm: {total_norm}")
            writer.add_scalar("train/norm", total_norm, global_step=self.state.global_step)

    @torch.no_grad()
    def _write_loss_and_learning_rate(
            self,
            writer: SummaryWriter,
            loss: torch.Tensor,
            loss_meta: Optional[Dict[str, torch.Tensor]] = None,
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
            if loss_meta is not None:
                # NOTICE: The keys in meta loss dict must be the same across all ranks to avoid any collective error
                for k in sorted(loss_meta.keys()):
                    dist.all_reduce(loss_meta[k])
                    loss_meta[k] /= dist.get_world_size()

        writer.add_scalar("train/loss", loss.detach().cpu().float(), global_step=self.state.global_step)
        if loss_meta is not None:
            loss_meta = {k: v.detach().cpu().float() for k, v in loss_meta.items()}
            writer.add_scalars("train/loss_meta", loss_meta, global_step=self.state.global_step)

        learning_rate = [group["lr"] if self.scheduler is None else group for group in
                         (self.optimizer.param_groups if self.scheduler is None else self.scheduler.get_last_lr())]
        writer.add_scalars("train/lr", {f"param_group_{i}": lr for i, lr in enumerate(learning_rate)},
                           global_step=self.state.global_step)

        logger.debug(
            f"Step: %s | Loss: %s | Learning rate: %s",
            self.state.global_step, loss.detach().cpu().numpy(),
            learning_rate[0] if len(learning_rate) == 1 else learning_rate
        )

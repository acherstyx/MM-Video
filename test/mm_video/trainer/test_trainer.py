# -*- coding: utf-8 -*-
# @Time    : 2024/3/6
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : test_trainer.py
import logging
import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from mm_video.modeling.meter import DummyMeter
from mm_video.trainer.trainer import *
from mm_video.trainer.trainer_utils import manual_seed
from mm_video.trainer.training_configs import *
from mm_video.utils.common.path import DisplayablePath

BATCH_SIZE = 16
SAVE_STEPS = 100


# Test dataset
class MNISTDataset(Dataset):
    def __init__(self, split: str):
        assert split in ["train", "test", "eval"]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = datasets.MNIST(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"),
            train=True if split == "train" else False, download=True, transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# Test model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, inputs):
        x, y = inputs
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        loss = F.nll_loss(output, y)
        return {'loss': loss, 'output': output}


def run_train(rank, world_size, output_dir, resume_from_checkpoint):
    logging.basicConfig(level=logging.DEBUG)

    dist.init_process_group(backend="nccl", init_method="tcp://localhost:2222", world_size=world_size, rank=rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    training_cfg = TrainingConfig(
        num_train_epochs=3, save_steps=SAVE_STEPS, logging_steps=1,
        resume_from_checkpoint=resume_from_checkpoint
    )
    dataloader_cfg = DataLoaderConfig(
        num_workers=0, train_batch_size=BATCH_SIZE, eval_batch_size=BATCH_SIZE, drop_last=True
    )
    training_strategy_cfg = TrainingStrategyConfig(strategy=TrainingStrategy.DDP)
    debug_cfg = DebugConfig(
        enable=True, max_train_steps=9999, max_eval_steps=9999, max_test_steps=9999,
        save_inputs=True, save_inputs_for_each_step=True
    )

    manual_seed(1)
    trainer = Trainer(
        model=CNN(),
        train_dataset=MNISTDataset(split="train"),
        eval_dataset=MNISTDataset(split="test"),
        meter=DummyMeter(),
        output_dir=output_dir,
        training=training_cfg,
        dataloader=dataloader_cfg,
        training_strategy=training_strategy_cfg,
        debug=debug_cfg
    )
    trainer.train()
    paths = DisplayablePath.make_tree(Path(output_dir))
    if dist.get_rank() == 0:
        for path in paths:
            print(path.displayable())


class TestTrainerResume(unittest.TestCase):

    @unittest.skipIf(not torch.cuda.is_available(), "GPU is required to run this test")
    def test_resume_from_checkpoint_distributed(self):
        world_size = 4
        len_train_dataset = len(MNISTDataset(split="train")) // world_size // BATCH_SIZE
        load_steps = int((len_train_dataset * 1.5)) // SAVE_STEPS * SAVE_STEPS
        last_checkpoint_step = int((len_train_dataset * 2)) // SAVE_STEPS * SAVE_STEPS
        with tempfile.TemporaryDirectory() as tmpdir:
            mp.spawn(run_train, args=(world_size, os.path.join(tmpdir, "train_1"), None), nprocs=world_size)
            mp.spawn(
                run_train,
                args=(world_size,
                      os.path.join(tmpdir, "train_2"),
                      os.path.join(tmpdir, "train_1", f"checkpoint-{load_steps}")),
                nprocs=world_size
            )

            # Check if model inputs are the same
            for inputs_filename in os.listdir(os.path.join(tmpdir, "train_2", "inputs")):
                input_1 = torch.load(os.path.join(tmpdir, "train_1", "inputs", inputs_filename))
                input_2 = torch.load(os.path.join(tmpdir, "train_2", "inputs", inputs_filename))
                self.assertTrue(torch.equal(input_1[0], input_2[0]), msg=inputs_filename)
                self.assertTrue(torch.equal(input_1[1], input_2[1]), msg=inputs_filename)

            # load state from last checkpoint
            state_before = TrainerState.load_from_json(
                os.path.join(tmpdir, "train_1", f"checkpoint-{last_checkpoint_step}", "trainer_state.json"))
            state_after = TrainerState.load_from_json(
                os.path.join(tmpdir, "train_2", f"checkpoint-{last_checkpoint_step}", "trainer_state.json"))

            # Checking loss
            for log_before, log_after in zip(state_before.log_history, state_after.log_history):
                self.assertEqual(log_before["step"], log_after["step"])
                self.assertEqual(log_before["loss"], log_after["loss"],
                                 msg=f'Log not the same for step {log_before["step"]}. Reusme: {load_steps}')

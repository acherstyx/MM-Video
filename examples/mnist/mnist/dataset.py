# -*- coding: utf-8 -*-
# @Time    : 2024/3/2
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : dataset.py

import os
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from mm_video.config import dataset_store


@dataset_store()
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

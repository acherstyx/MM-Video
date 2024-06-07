# -*- coding: utf-8 -*-
# @Time    : 4/29/24
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : test_transformers.py
import unittest

from omegaconf import OmegaConf
from rich import print

from mm_video.config.compat import run_post_init
from mm_video.config.compat.transformers import *


class TestTransformers(unittest.TestCase):
    def test_training_arguments(self):
        cfg = OmegaConf.structured(TrainingArguments)
        cfg.output_dir = ""
        print(OmegaConf.to_yaml(cfg))
        print(run_post_init(cfg))


if __name__ == '__main__':
    unittest.main()

# -*- coding: utf-8 -*-
# @Time    : 4/29/24
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : test_huggingface.py
import unittest

from omegaconf import OmegaConf
from rich import print
import tempfile

from mm_video.config.compat import run_post_init
from mm_video.config.compat.transformers import *
from mm_video.config.compat.peft import *


class TestPEFT(unittest.TestCase):
    def test_peft_config(self):
        cfg = OmegaConf.structured(PeftConfig)
        print(OmegaConf.to_yaml(cfg))

    def test_lora_config(self):
        cfg = OmegaConf.structured(LoraConfig)
        print(OmegaConf.to_yaml(cfg))
        print(run_post_init(cfg))


class TestTransformers(unittest.TestCase):
    def test_training_arguments(self):
        with tempfile.TemporaryDirectory() as output_dir:
            cfg = OmegaConf.structured(TrainingArguments)
            cfg.output_dir = output_dir
            print(OmegaConf.to_yaml(cfg))
            print(run_post_init(cfg))


if __name__ == '__main__':
    unittest.main()

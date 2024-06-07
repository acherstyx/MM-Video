# -*- coding: utf-8 -*-
# @Time    : 5/18/24
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : test_peft.py

import unittest

from omegaconf import OmegaConf
from rich import print

from mm_video.config.compat import run_post_init
from mm_video.config.compat.peft import PeftConfig, LoraConfig


class TestPEFT(unittest.TestCase):
    def test_peft_config(self):
        cfg = OmegaConf.structured(PeftConfig)
        print(OmegaConf.to_yaml(cfg))

    def test_lora_config(self):
        cfg = OmegaConf.structured(LoraConfig)
        print(OmegaConf.to_yaml(cfg))
        print(run_post_init(cfg))


if __name__ == '__main__':
    unittest.main()

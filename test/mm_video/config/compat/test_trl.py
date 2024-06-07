# -*- coding: utf-8 -*-
# @Time    : 6/8/24
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : test_trl.py

import unittest
import tempfile

from omegaconf import OmegaConf

from mm_video.config.compat.trl import *
from mm_video.config.compat import run_post_init


class TestTRL(unittest.TestCase):
    def test_dpo_config(self):
        with tempfile.TemporaryDirectory() as output_dir:
            cfg = OmegaConf.structured(DPOConfig)
            cfg.output_dir = output_dir
            print(OmegaConf.to_yaml(cfg))
            print(run_post_init(cfg))


if __name__ == '__main__':
    unittest.main()

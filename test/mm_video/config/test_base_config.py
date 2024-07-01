# -*- coding: utf-8 -*-
# @Time    : 6/30/24
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : test_base_config.py

from hydra import initialize
from hydra.core.global_hydra import GlobalHydra


def test_discovery() -> None:
    with initialize(version_base=None):
        config_loader = GlobalHydra.instance().config_loader()
        assert "_mm_video_template" in config_loader.get_group_options("")

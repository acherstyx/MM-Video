# -*- coding: utf-8 -*-
# @Time    : 6/30/24
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : test_mm_video.py

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.core.plugins import Plugins, SearchPathPlugin
from hydra_plugins.mm_video.mm_video import MMVideoSearchPathPlugin
from omegaconf import OmegaConf


def test_discovery() -> None:
    assert MMVideoSearchPathPlugin.__name__ in [
        x.__name__ for x in Plugins.instance().discover(SearchPathPlugin)
    ]


def test_config_installed() -> None:
    with initialize(version_base=None):
        config_loader = GlobalHydra.instance().config_loader()
        assert "mm_video_template" in config_loader.get_group_options("")
        assert "mm_video_default_log" in config_loader.get_group_options("hydra/job_logging")
        assert "mm_video_distributed_log" in config_loader.get_group_options("hydra/job_logging")


def test_build_config() -> None:
    with initialize(version_base=None):
        cfg = compose(config_name="mm_video_template")
        print(OmegaConf.to_yaml(cfg))

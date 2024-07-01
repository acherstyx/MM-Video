# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class MMVideoSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Import config to add config group to hydra
        import mm_video.config

        # Appends the search path for this plugin to the end of the search path
        search_path.append("mm-video", "pkg://hydra_plugins.mm_video.configs")

# -*- coding: utf-8 -*-
# @Time    : 7/12/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : profile.py

import time
import logging
import numpy as np
import torch
from collections import defaultdict, deque
import functools
from tabulate import tabulate
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


def format_time(s: float) -> str:
    """Return a nice string representation of `s` seconds."""
    m = int(s / 60)
    s -= m * 60
    h = int(m / 60)
    m -= h * 60
    ms = int(s * 100000) / 100
    s = int(s * 100) / 100.0
    return ("" if h == 0 else str(h) + "h") + ("" if m == 0 else str(m) + "m") + ("" if s == 0 else str(s) + "s") + \
        (str(ms) + "ms" if s == 0 else "")


class Timer(object):
    start: float
    last_checkpoint: float
    time_history: Dict[str, deque]

    def __init__(
            self,
            msg="",
            synchronize: bool = False,
            history_size: int = 1000,
            precision: int = 3,
            print_func: Callable = logger.debug
    ):
        """
        Init and start timer
        :param msg: Message to print before timing
        :param synchronize: Call `torch.cuda.synchronize()` when getting time
        :param history_size:
        :param precision: round seconds to a given precision in decimal digits to avoid verbose
        """
        self.msg = msg
        self.synchronize = synchronize
        self.precision = precision
        self.print_func = print_func
        self.history_size = history_size

        self.reset()

        if self.msg:
            self.print_func(msg)

    @property
    def averaged_time_history(self) -> Dict[str, float]:
        return {name: np.mean(list(duration)).item() for name, duration in self.time_history.items()}

    def _get_time(self) -> float:
        """
        call `torch.cuda.synchronize()` and return rounded time in seconds
        :return: current time in seconds
        """
        if self.synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()
        return round(time.time(), self.precision)

    def reset(self):
        """
        reset to the init status, restart timing
        :return:
        """
        self.start = self._get_time()
        self.last_checkpoint = self._get_time()
        self.time_history = defaultdict(functools.partial(deque, maxlen=self.history_size))

    def lap(self, name: Optional[str] = None):
        if name is None:
            name = f"Lap No. {len(self.time_history) + 1}"
        current_time = self._get_time()
        duration = (current_time - self.last_checkpoint)
        self.last_checkpoint = current_time
        self.time_history[name].append(duration)

    def end(self):
        duration = self._get_time() - self.start
        if self.msg:
            self.print_func(f"{self.msg} [took {format_time(duration)}]")

    def __enter__(self):
        self.start = self._get_time()
        return self

    def __exit__(self, typ, value, traceback):
        self.end()

    def __call__(self, name: Optional[str] = None):
        return self.lap(name=name)

    def get_info(self, averaged=True):
        return {
            k: round(float(np.mean(v)), self.precision) if averaged else round(v[-1], self.precision)
            for k, v in self.time_history.items()
        }

    def __str__(self):
        return str(self.get_info())

    def print(self):
        data = [[name, format_time(duration)] for name, duration in self.averaged_time_history.items()]
        print(tabulate(data, headers=["Stage", "Time (ms)"], tablefmt="simple"))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    with Timer("Run as context manager...") as f:
        time.sleep(1.12)

    timer = Timer("Run as function call...")
    time.sleep(0.5)
    timer.lap()
    time.sleep(0.21)
    timer.lap()
    timer.end()
    timer.print()
    print(timer)
    print(timer.get_info())

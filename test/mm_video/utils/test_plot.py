# -*- coding: utf-8 -*-
# @Time    : 6/7/24
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : test_plot.py

import unittest

import numpy as np
import matplotlib.pyplot as plt

from mm_video.utils.plot import *


class TestShowDistribution(unittest.TestCase):
    def test_show_distribution(self):
        fig = show_distribution(distribution=np.random.normal(1, size=(10000,)))
        fig.show()


if __name__ == '__main__':
    unittest.main()

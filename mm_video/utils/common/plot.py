# -*- coding: utf-8 -*-
# @Time    : 2024/3/1
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : plot.py

import io
from typing import Union, List, Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import ToTensor


def fig_to_image(fig_to_save: plt.Figure, **savefig_kwargs) -> Image:
    buf = io.BytesIO()
    fig_to_save.savefig(buf, format="png", **savefig_kwargs)
    fig_to_save.clear()
    buf.seek(0)
    return Image.open(buf)


def fig_to_tensor(fig_to_save: plt.Figure, **savefig_kwargs) -> torch.Tensor:
    return ToTensor()(fig_to_image(fig_to_save, **savefig_kwargs))


def show_distribution(
        distribution: Union[List[float], Any],
        bins: int = 100
) -> plt.Figure:
    """
    Show histogram with kernel density estimation (KDE).
    Call plt.show() or plt.savefig() after calling this function to view the result.
    Based on: https://stackoverflow.com/questions/66151692/pyplot-draw-a-smooth-curve-over-a-histogram

    :param distribution: A list of integers/floats or a numpy array
    :param bins:
    :return:
    """
    with plt.style.context("ggplot"):
        fig, ax = plt.subplots(dpi=300)
        df = pd.DataFrame(dict(variable=distribution))
        df['variable'].plot.hist(bins=bins, density=True, alpha=0.4, ax=ax)
        ax.set_xlim(ax.get_xlim())
        df['variable'].plot.density(color='r', alpha=1.0, linewidth=2.0, ax=ax)
        fig.tight_layout()
    return fig

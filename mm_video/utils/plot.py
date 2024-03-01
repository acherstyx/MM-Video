# -*- coding: utf-8 -*-
# @Time    : 2024/3/1
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : plot.py

import io
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToTensor


def fig_to_image(fig_to_save: plt.Figure, **savefig_kwargs) -> Image:
    buf = io.BytesIO()
    fig_to_save.savefig(buf, format="png", **savefig_kwargs)
    fig_to_save.clear()
    buf.seek(0)
    return Image.open(buf)


def fig_to_tensor(fig_to_save: plt.Figure, **savefig_kwargs) -> torch.Tensor:
    return ToTensor()(fig_to_image(fig_to_save, **savefig_kwargs))

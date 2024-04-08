# -*- coding: utf-8 -*-
# @Time    : 2022/11/13 01:50
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : meter.py


import logging
import torch
from torch.utils.tensorboard import SummaryWriter

from abc import ABC, abstractmethod
from typing import Any, Dict, Union, Optional

__all__ = ["Meter", "DummyMeter"]

logger = logging.getLogger(__name__)


class Meter(ABC):
    """
    Base class for Meters
    """

    @abstractmethod
    def update(
            self, inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor],
            writer: Optional[SummaryWriter], main_tag: str, global_step: int
    ) -> Optional[float]:
        """
        call on each step, update inner status based on the inputs and outputs of model.
        return a dict of scalars, which will be written to the tensorboard by the trainer by default.
        :param inputs: the dataloader outputs/model inputs
        :param outputs: the model outputs
        :param writer:
        :param main_tag:
        :param global_step:
        """

    @abstractmethod
    def summary(self, writer: Optional[SummaryWriter], main_tag: str, global_step: int) -> Optional[Dict[str, float]]:
        """call at the end of the train/test/val epoch.
        return a dict of summarized metrics of current epoch, which will be written to the tensorboard."""

    @abstractmethod
    def reset(self) -> None:
        """reset to initial status"""

    @staticmethod
    def write_structured_scalar(
            writer: Optional[SummaryWriter],
            nested_scalars: Optional[Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]],
            main_tag: str,
            global_step: int
    ):
        if writer is not None and nested_scalars is not None:
            assert type(nested_scalars) is dict
            for tag, scalar in nested_scalars.items():
                if type(scalar) == dict:
                    writer.add_scalars(f"{main_tag}/{tag}", scalar, global_step=global_step)
                else:
                    writer.add_scalar(f"{main_tag}/{tag}", scalar, global_step=global_step)


class DummyMeter(Meter):

    def update(
            self, inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor],
            writer: SummaryWriter, main_tag: str, global_step: int
    ) -> Dict[str, Any]:
        pass

    def summary(self, writer: SummaryWriter, main_tag: str, global_step: int) -> Dict[str, Any]:
        pass

    def reset(self) -> None:
        pass

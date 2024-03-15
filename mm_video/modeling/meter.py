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

__all__ = ["Meter", "DummyMeter", "GroupMeter"]

logger = logging.getLogger(__name__)


class Meter(ABC):
    """
    Base class for Meters
    """

    @abstractmethod
    def update(
            self,
            inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor],
            writer: Optional[SummaryWriter], main_tag: str, global_step: int
    ) -> Optional[Dict[str, float]]:
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


class GroupMeter(Meter):
    def __init__(self, meters: Dict[str, Meter]):
        self.meters = meters

    @staticmethod
    def merge_metrics(
            metrics: Dict[str, float],
            meter_name: str,
            meter_metric: Union[Dict[str, float], None]
    ):
        """This method update metrics IN-PLACE. Add meter_name as prefix."""
        if meter_metric is not None and type(meter_metric) is dict:
            for key, value in meter_metric.items():
                metrics[f"{meter_name}_{key}"] = value
        elif meter_metric is not None and not type(meter_metric) is dict:
            logger.warning(f"A return type of dictionary or None value is required for update/summarize method of "
                           f"meter, but received {type(meter_metric).__name__}. The returned metrics will be ignored.")

    def update(
            self,
            inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor],
            writer: Optional[SummaryWriter], main_tag: str, global_step: int
    ) -> Optional[Dict[str, float]]:
        metrics = {}
        for meter_name, meter in self.meters.items():
            meter_metrics = meter.update(inputs, outputs, writer, main_tag, global_step)
            self.merge_metrics(metrics, meter_name, meter_metrics)
        return metrics

    def summary(self, writer: Optional[SummaryWriter], main_tag: str, global_step: int) -> Optional[Dict[str, float]]:
        metrics = {}
        for meter_name, meter in self.meters.items():
            meter_metrics = meter.summary(writer=writer, main_tag=main_tag, global_step=global_step)
            self.merge_metrics(metrics, meter_name, meter_metrics)
        return metrics

    def reset(self) -> None:
        for meter_name, meter in self.meters.items():
            meter.reset()


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

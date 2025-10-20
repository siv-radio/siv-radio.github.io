# Copyright (C) 2025 Igor Sivchek
# Licensed under the MIT License.
# See license text at [https://opensource.org/license/mit].

"""
Utilities of different kinds.
"""

import cProfile
from collections.abc import Iterable
import pstats
from typing import Any, Callable, Optional, Union

import timm
import torch

__all__ = ["NativeScalerV2", "Profile", "check_cuda_support", "get_device"]


def check_cuda_support(*, inform: bool = False) -> tuple[bool, bool]:
    has_cuda = False
    has_bf16 = False
    if torch.cuda.is_available():
        has_cuda = True
        if torch.cuda.is_bf16_supported():
            has_bf16 = True
    print(f"CUDA support: {has_cuda}")
    print(f"BF16 support: {has_bf16}")
    return (has_cuda, has_bf16)


def get_device(*, force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    else:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Mixed precision training may cause "UserWarning: Detected call of
# `lr_scheduler.step()` before `optimizer.step()`" issue. A special technique
# should be used to avoid this problem. See
# https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradsclaer/92930
# https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step/164814
# https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step-in-pytorch-1-1-0-and-later-you-should-call-them-in-the-opposite-order-optimizer-step-before-lr-scheduler-step/88295
# Basic resources
# https://docs.pytorch.org/docs/2.7/amp.html
# https://github.com/pytorch/pytorch/blob/v2.7.0/torch/amp/grad_scaler.py
# https://github.com/huggingface/pytorch-image-models/blob/v1.0.15/timm/utils/cuda.py
# It seems, ideally, there should be retraining on a problematic batch, not
# just skipping of a learning rate step. There should be a loop to find an
# applicable scale. Of course, the number of cycles should be limited.
class NativeScalerV2(timm.utils.cuda.NativeScaler):
    def __init__(self, device: str = "cuda") -> None:
        super().__init__(device=device)
        self._stepped: bool = False

    def __call__(
        self,
        *,
        loss: Union[torch.Tensor, Iterable[torch.Tensor]],
        optimizer: torch.optim.Optimizer,
        clip_grad: Optional[float] = None,
        clip_mode: str = "norm",
        parameters: Optional[Iterable] = None,
        create_graph: bool = False,
        need_update: bool = True,
    ) -> None:
        scale = self.get_scale()
        super().__call__(
            loss=loss, optimizer=optimizer, clip_grad=clip_grad,
            clip_mode=clip_mode, parameters=parameters,
            create_graph=create_graph, need_update=need_update,
        )
        self._stepped = (scale <= self.get_scale())

    def has_stepped(self) -> bool:
        return self._stepped

    def get_scale(self) -> float:
        return self._scaler.get_scale()


# A context manager to wrap a ``cProfile.Profile`` object. It can be activated
# or deactivated without the necessity to change code containing it.
# https://docs.python.org/3/library/profile.html
# https://stackoverflow.com/questions/29630667/how-can-i-analyze-a-file-created-with-pstats-dump-statsfilename-off-line
class Profile:
    def __init__(self, *, active: bool = True):
        self.obj = cProfile.Profile()
        self.__active = active

    def __enter__(self):
        self.obj.enable()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.obj.disable()
        if exc_type is not None:
            return False

    def __copy__(self):
        raise TypeError("A 'copy' operation is not supported.")

    def __deepcopy__(self):
        raise TypeError("A 'deepcopy' operation is not supported.")

    def dump(
        self,
        *,
        filename: str,
        sort_by: str = "stdname",
        strip_dirs: bool = False,
    ):
        if self.__active:
            with open(file=(filename + ".log"), mode="w") as out_stream:
                stats = pstats.Stats(self.obj, stream=out_stream)
                if strip_dirs:
                    stats.strip_dirs()
                stats.sort_stats(sort_by)
                stats.dump_stats(filename + ".prof")
                stats.print_stats()

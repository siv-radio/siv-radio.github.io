# Copyright (C) 2025 Igor Sivchek
# Licensed under the MIT License.
# See license text at [https://opensource.org/license/mit].

"""
Some visualization tools.
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import matplotlib
import numpy as np
import torch

__all__ = ["draw_cumulative_accuracy", "show_images"]


def draw_image(
    *,
    ax: matplotlib.axis.Axis,
    image: torch.Tensor,
    mean: np.ndarray,
    std: np.ndarray,
    title: Optional[str] = None,
) -> None:
    image = image.numpy().transpose((1, 2, 0))
    # This procedure is unnecessary if data is not transformed.
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    if title is not None:
        ax.set_title(title)


def show_images(
    *,
    inputs: torch.Tensor,
    targets: list[str],
    predictions: list[str],
    mean: Sequence[float],
    std: Sequence[float],
    figsize: tuple[int] = (8, 8),
) -> None:
    inputs = inputs.detach().cpu()
    mean = np.array(mean)
    std = np.array(std)
    num_rows = 2
    num_cols = 3
    num_plots = num_rows * num_cols
    fig, axes = matplotlib.pyplot.subplots(
        nrows=num_rows, ncols=num_cols,
        figsize=figsize, constrained_layout=True
    )
    plot_idx = 0
    for input_idx, (image, target, prediction) in (
        enumerate(zip(inputs, targets, predictions))
    ):
        if plot_idx >= num_plots:
            break
        row_idx = input_idx // num_cols
        col_idx = input_idx % num_cols
        title = "targ.: {0}\npred.: {1}".format(target, prediction)
        ax = axes[row_idx, col_idx]
        ax.axis("off")
        draw_image(ax=ax, image=image, mean=mean, std=std, title=title)
        plot_idx += 1


def draw_cumulative_accuracy(
    *,
    shares: np.ndarray,
    cum_accs: np.ndarray,
    figsize: tuple[int] = (8, 8),
) -> None:
    fig, ax = matplotlib.pyplot.subplots(
        nrows=1, ncols=1, figsize=figsize, constrained_layout=True
    )
    fig.suptitle("Cumulative accuracy function")
    ax.plot(shares, cum_accs)
    ax.set_xlabel("share")
    ax.set_ylabel("accuracy", rotation="horizontal")
    ax.yaxis.set_label_coords(0.0, 1.05)
    ax.grid()
    fig.show()
